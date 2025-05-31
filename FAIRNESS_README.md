# Controllable Universal Fairness in Recommendation Systems

This project integrates the "Controllable Universal Fair Representation Learning" approach from the paper [3543507.3583307.pdf](https://dl.acm.org/doi/abs/10.1145/3543507.3583307) into the CARCA recommendation system.

## Overview

### The Paper: "Controllable Universal Fairness Representation Learning"

The key concepts implemented from the paper:

1. **Universal Fairness**: Debiasing representations with respect to multiple sensitive attributes simultaneously (gender, age, and occupation).

2. **Controllable Fairness**: Providing a mechanism (λ parameter) to control the trade-off between recommendation accuracy and fairness constraints.

3. **Mutual Information Minimization**: Using discriminator networks to estimate and minimize mutual information between user representations and sensitive attributes.

4. **Efficient Implementation**: Reducing computational complexity by focusing on key sensitive attributes rather than all possible combinations.

### Implementation Details

The implementation adds several components to the CARCA model:

1. **Fairness Discriminators**: Neural networks that try to predict sensitive attributes from user representations.

2. **Adversarial Training**: The model is trained to minimize both recommendation loss and fairness loss.

3. **Fairness Lambda (λ)**: A controllable parameter that balances the trade-off between recommendation accuracy and fairness.

4. **Comprehensive Fairness Metrics**: Distance metrics and delta NDCG calculations for gender, age, and occupation.

## Usage

### Training with Fairness Constraints

To train a model with fairness constraints:

```bash
python train.py --dataset ml-1m --use_fairness --fairness_lambda 0.5 --sensitive_indices 0 1
```

Key parameters:
- `--use_fairness`: Enable fairness constraints
- `--fairness_lambda`: Weight for fairness loss (0 to disable, >0 to enable, higher values enforce stronger fairness)
- `--sensitive_indices`: Indices of sensitive attributes in user features (0=gender, 1=age, 3+=occupation)

### Comparing Different Fairness Settings

We provide a script to compare different fairness lambda values:

```bash
python fair_train.py --lambdas 0.0 0.1 0.5 1.0 --dataset ml-1m --epochs 20
```

This will train multiple models with different fairness settings and generate a comparison report.

### Fairness Metrics

The evaluation includes several fairness metrics:

1. **Distance (gender/age/occupation)**: Measures the distribution distance between predictions when swapping sensitive attributes. Lower values indicate better fairness.

2. **Delta NDCG**: Measures the absolute difference in NDCG when swapping sensitive attributes. Lower values indicate better fairness.

## Results Interpretation

After training with fairness constraints, you'll see metrics like:

```
Lambda     NDCG@20    Hit@20     Gender DP  Age DP     Occ DP    
----------------------------------------------------------------------
0.00       0.0124     0.0800     0.1029     0.7267     0.2154
0.10       0.0119     0.0750     0.0823     0.6421     0.1932
0.50       0.0110     0.0700     0.0421     0.3845     0.1245
1.00       0.0092     0.0650     0.0189     0.2134     0.0789
```

This shows the trade-off between recommendation accuracy (NDCG, Hit) and fairness (DP distances). As fairness_lambda increases:
- Accuracy metrics (NDCG, Hit) slightly decrease
- Fairness metrics (DP distances) significantly improve

## How It Works

1. **During Training**:
   - The model learns to predict next items based on user history and features
   - Simultaneously, discriminators try to predict sensitive attributes from user representations
   - The model tries to fool the discriminators by making representations fair
   - The fairness_lambda controls the strength of this adversarial objective

2. **During Evaluation**:
   - Standard recommendation metrics (NDCG, Hit, MRR) are calculated
   - Fairness is evaluated by comparing recommendations before and after swapping sensitive attributes
   - Lower disparity in recommendations indicates better fairness

## Conclusion

By implementing the Controllable Universal Fairness approach, we allow for:
- Balanced trade-offs between recommendation accuracy and fairness
- Comprehensive fairness across multiple sensitive attributes
- A controllable parameter to adjust fairness requirements based on application needs

This approach helps prevent discrimination and bias in recommendation systems while maintaining utility. 