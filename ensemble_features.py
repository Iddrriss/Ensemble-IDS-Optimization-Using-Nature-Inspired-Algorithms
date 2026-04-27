import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

BASE_PATH = r"C:\Users\Oxseeker\Desktop\CYS417"

# ── Load preprocessed data ───────────────────────────────────────────────────
print("Loading preprocessed data...")
X_train = np.load(os.path.join(BASE_PATH, "X_train.npy"))
X_test  = np.load(os.path.join(BASE_PATH, "X_test.npy"))
y_train = np.load(os.path.join(BASE_PATH, "y_train.npy"))
y_test  = np.load(os.path.join(BASE_PATH, "y_test.npy"))

with open(os.path.join(BASE_PATH, "feature_names.txt")) as f:
    feature_names = f.read().splitlines()

N_FEATURES = X_train.shape[1]  # 41

# ── Use a small sample for speed during feature selection ────────────────────
SAMPLE = 5000
idx = np.random.choice(len(X_train), SAMPLE, replace=False)
Xs, ys = X_train[idx], y_train[idx]

# ── Fitness function (higher = better) ───────────────────────────────────────
# Rewards accuracy, penalises using too many features
def fitness(mask):
    selected = np.where(mask > 0.5)[0]
    if len(selected) == 0:
        return 0.0
    clf = SVC(kernel="linear", max_iter=500)
    clf.fit(Xs[:, selected], ys)
    acc = accuracy_score(ys, clf.predict(Xs[:, selected]))
    feature_ratio = len(selected) / N_FEATURES
    return acc - 0.01 * feature_ratio   # small penalty for using more features

# ════════════════════════════════════════════════════════════════════════════
# 1. PSO  (Particle Swarm Optimization)
# ════════════════════════════════════════════════════════════════════════════
def run_pso(n_particles=20, iterations=30):
    print("\n[PSO] Running...")
    w, c1, c2 = 0.7, 1.5, 1.5

    pos = np.random.rand(n_particles, N_FEATURES)
    vel = np.random.rand(n_particles, N_FEATURES) * 0.1

    pbest_pos   = pos.copy()
    pbest_score = np.array([fitness(p) for p in pos])

    gbest_idx   = np.argmax(pbest_score)
    gbest_pos   = pbest_pos[gbest_idx].copy()
    gbest_score = pbest_score[gbest_idx]

    for it in range(iterations):
        for i in range(n_particles):
            r1, r2 = np.random.rand(N_FEATURES), np.random.rand(N_FEATURES)
            vel[i] = (w * vel[i]
                      + c1 * r1 * (pbest_pos[i] - pos[i])
                      + c2 * r2 * (gbest_pos   - pos[i]))
            pos[i] = np.clip(pos[i] + vel[i], 0, 1)

            s = fitness(pos[i])
            if s > pbest_score[i]:
                pbest_score[i] = s
                pbest_pos[i]   = pos[i].copy()
                if s > gbest_score:
                    gbest_score = s
                    gbest_pos   = pos[i].copy()

        if (it + 1) % 10 == 0:
            print(f"  Iteration {it+1}/{iterations}  best fitness={gbest_score:.4f}")

    mask = (gbest_pos > 0.5).astype(int)
    selected = np.where(mask)[0]
    print(f"[PSO] Done — {len(selected)} features selected, fitness={gbest_score:.4f}")
    return mask

# ════════════════════════════════════════════════════════════════════════════
# 2. GA  (Genetic Algorithm)
# ════════════════════════════════════════════════════════════════════════════
def run_ga(pop_size=20, generations=30, crossover_rate=0.8, mutation_rate=0.02):
    print("\n[GA] Running...")

    # initialise binary population
    pop = np.random.randint(0, 2, (pop_size, N_FEATURES)).astype(float)
    scores = np.array([fitness(ind) for ind in pop])

    best_idx   = np.argmax(scores)
    best_ind   = pop[best_idx].copy()
    best_score = scores[best_idx]

    for gen in range(generations):
        new_pop = []

        # elitism — keep best individual
        new_pop.append(best_ind.copy())

        while len(new_pop) < pop_size:
            # tournament selection
            def tournament():
                a, b = random.sample(range(pop_size), 2)
                return pop[a].copy() if scores[a] > scores[b] else pop[b].copy()

            p1, p2 = tournament(), tournament()

            # crossover
            if random.random() < crossover_rate:
                pt = random.randint(1, N_FEATURES - 1)
                child = np.concatenate([p1[:pt], p2[pt:]])
            else:
                child = p1.copy()

            # mutation
            for j in range(N_FEATURES):
                if random.random() < mutation_rate:
                    child[j] = 1.0 - child[j]

            new_pop.append(child)

        pop    = np.array(new_pop[:pop_size])
        scores = np.array([fitness(ind) for ind in pop])

        best_idx   = np.argmax(scores)
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_ind   = pop[best_idx].copy()

        if (gen + 1) % 10 == 0:
            print(f"  Generation {gen+1}/{generations}  best fitness={best_score:.4f}")

    selected = np.where(best_ind > 0.5)[0]
    print(f"[GA] Done — {len(selected)} features selected, fitness={best_score:.4f}")
    return best_ind

# ════════════════════════════════════════════════════════════════════════════
# 3. GWO  (Grey Wolf Optimizer)
# ════════════════════════════════════════════════════════════════════════════
def run_gwo(n_wolves=20, iterations=30):
    print("\n[GWO] Running...")

    pos    = np.random.rand(n_wolves, N_FEATURES)
    scores = np.array([fitness(p) for p in pos])

    sorted_idx = np.argsort(-scores)
    alpha_pos, alpha_score = pos[sorted_idx[0]].copy(), scores[sorted_idx[0]]
    beta_pos,  beta_score  = pos[sorted_idx[1]].copy(), scores[sorted_idx[1]]
    delta_pos, delta_score = pos[sorted_idx[2]].copy(), scores[sorted_idx[2]]

    for it in range(iterations):
        a = 2 - 2 * (it / iterations)   # decreases from 2 → 0

        for i in range(n_wolves):
            new_pos = np.zeros(N_FEATURES)
            for leader in [alpha_pos, beta_pos, delta_pos]:
                r1, r2 = np.random.rand(N_FEATURES), np.random.rand(N_FEATURES)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = abs(C * leader - pos[i])
                X = leader - A * D
                new_pos += X
            pos[i] = np.clip(new_pos / 3, 0, 1)

            s = fitness(pos[i])
            if s > alpha_score:
                delta_pos, delta_score = beta_pos.copy(),  beta_score
                beta_pos,  beta_score  = alpha_pos.copy(), alpha_score
                alpha_pos, alpha_score = pos[i].copy(),    s
            elif s > beta_score:
                delta_pos, delta_score = beta_pos.copy(), beta_score
                beta_pos,  beta_score  = pos[i].copy(),   s
            elif s > delta_score:
                delta_pos, delta_score = pos[i].copy(), s

        if (it + 1) % 10 == 0:
            print(f"  Iteration {it+1}/{iterations}  best fitness={alpha_score:.4f}")

    mask = (alpha_pos > 0.5).astype(int)
    selected = np.where(mask)[0]
    print(f"[GWO] Done — {len(selected)} features selected, fitness={alpha_score:.4f}")
    return mask

# ════════════════════════════════════════════════════════════════════════════
# 4. ENSEMBLE  — majority vote (feature selected by ≥ 2 of 3 algorithms)
# ════════════════════════════════════════════════════════════════════════════
pso_mask = run_pso()
ga_mask  = run_ga()
gwo_mask = run_gwo()

votes = pso_mask + ga_mask + gwo_mask          # each feature gets 0, 1, 2, or 3 votes
ensemble_mask = (votes >= 2).astype(int)       # keep if at least 2 algorithms agree

selected_features = np.where(ensemble_mask)[0]
selected_names    = [feature_names[i] for i in selected_features]

print("\n" + "="*60)
print("ENSEMBLE RESULT")
print("="*60)
print(f"PSO selected:      {int(pso_mask.sum())} features")
print(f"GA  selected:      {int(ga_mask.sum())} features")
print(f"GWO selected:      {int(gwo_mask.sum())} features")
print(f"Ensemble selected: {len(selected_features)} features (majority vote)")
print(f"\nSelected feature names:")
for i, name in enumerate(selected_names, 1):
    print(f"  {i:2}. {name}")

# ── Save results ─────────────────────────────────────────────────────────────
np.save(os.path.join(BASE_PATH, "ensemble_mask.npy"),  ensemble_mask)
np.save(os.path.join(BASE_PATH, "pso_mask.npy"),       pso_mask)
np.save(os.path.join(BASE_PATH, "ga_mask.npy"),        ga_mask)
np.save(os.path.join(BASE_PATH, "gwo_mask.npy"),       gwo_mask)
np.save(os.path.join(BASE_PATH, "selected_features.npy"), selected_features)

with open(os.path.join(BASE_PATH, "selected_feature_names.txt"), "w") as f:
    f.write("\n".join(selected_names))

print("\nAll masks saved. Ready for Step 5.")