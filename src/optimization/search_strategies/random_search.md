```mermaid
classDiagram
    class BaseSearch {
        -config: dict
        -logger: Logger
    }
    class RandomSearchOptimizer {
        -config: dict
        -logger: Logger
        -param_distributions: dict
        -n_iter: int
        -cv: int
        -scoring: str
        +__init__(config, logger)
        -_generate_random_combinations() list
        -_cross_validate(model, params, X, y) float
        +optimize(model, X, y) Tuple[dict, float]
    }
    BaseSearch <|.. RandomSearchOptimizer : Inherits
```
```mermaid
sequenceDiagram
    participant C as Client
    participant R as RandomSearchOptimizer
    participant B as BaseModel
    participant L as Logger
    C->>R: optimize(model, X, y)
    R->>L: info("Starting Random Search...")
    R->>R: _generate_random_combinations()
    loop Over param_combinations
        R->>L: debug("Evaluating parameters: {...}")
        R->>R: _cross_validate(model, params, X, y)
        R->>B: type(model)(config, logger)
        R->>B: set_params(params)
        R->>B: fit(X_train_df, y_train_series)
        R->>B: predict(X_val_df)
        R->>R: Compute score
        R->>L: debug("Score for params {...}: {...}")
        R->>R: Update best_score, best_params if better
    end
    R->>L: debug("Best parameters: {...}, Best score: {...}")
    R->>L: info("Random Search completed.")
    R-->>C: Return (best_params, best_score)
```
```mermaid
stateDiagram-v2
    [*] --> Uninitialized
    Uninitialized --> Initialized : __init__(config, logger)
    Initialized --> Optimizing : optimize(model, X, y)
    Optimizing --> Completed : Best parameters and score determined
    Completed --> Initialized : Ready for another optimize call
    Completed --> [*]
```
```mermaid
graph TD
    A[Start Random Search] --> B[Sample 1: {param1: v_rand1, param2: w_rand1}]
    A --> C[Sample 2: {param1: v_rand2, param2: w_rand2}]
    A --> D[Sample N: {param1: v_randN, param2: w_randM}]
    B --> B1[Fold 1 Score]
    B --> B2[Fold 2 Score]
    B --> B3[...]
    C --> C1[Fold 1 Score]
    C --> C2[Fold 2 Score]
    C --> C3[...]
    D --> D1[Fold 1 Score]
    D --> D2[Fold 2 Score]
    D --> D3[...]
```
```mermaid
graph TD
    A[Client] -->|model, X, y| B[RandomSearchOptimizer]
    B -->|Log messages| C[Logger]
    B -->|param_combinations| B
    B -->|Clone, set_params, fit, predict| D[BaseModel]
    D -->|Predictions, scores| B
    B -->|best_params, best_score| A
```