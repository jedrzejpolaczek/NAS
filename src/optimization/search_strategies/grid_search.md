```mermaid
classDiagram
    class BaseSearch {
        -config: dict
        -logger: Logger
    }
    class GridSearchOptimizer {
        -config: dict
        -logger: Logger
        -param_grid: dict
        -cv: int
        -scoring: str
        +__init__(config, logger)
        -_generate_param_combinations() list
        -_cross_validate(model, params, X, y) float
        +optimize(model, X, y) Tuple[dict, float]
    }
    BaseSearch <|.. GridSearchOptimizer : Inherits
```
```mermaid
sequenceDiagram
    participant C as Client
    participant G as GridSearchOptimizer
    participant B as BaseModel
    participant L as Logger
    C->>G: optimize(model, X, y)
    G->>L: info("Starting Grid Search...")
    G->>G: _generate_param_combinations()
    loop Over param_combinations
        G->>G: _cross_validate(model, params, X, y)
        G->>B: type(model)(config, logger)
        G->>B: set_params(params)
        G->>B: fit(X_train_df, y_train_series)
        G->>B: predict(X_val_df)
        G->>G: Compute score
        G->>G: Update best_score, best_params if better
    end
    G->>L: debug("Best parameters: {...}, Best score: {...}")
    G->>L: info("Grid Search completed.")
    G-->>C: Return (best_params, best_score)
```
```mermaid
stateDiagram-v2
    [*] --> Uninitialized
    Uninitialized --> Initialized : __init__(config, logger)
    Initialized --> Optimizing : optimize(model, X, y)
    Optimizing --> Completed : Best parameters and score determined
    Completed --> Initialized : Can restart with new optimize call
    Completed --> [*]
```
```mermaid
graph TD
    A[Start Grid Search] --> B[Combo 1: {param1: v1, param2: w1}]
    A --> C[Combo 2: {param1: v1, param2: w2}]
    A --> D[Combo N: {param1: vN, param2: wM}]
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
    A[Client] -->|model, X, y| B[GridSearchOptimizer]
    B -->|Log messages| C[Logger]
    B -->|param_combinations| B
    B -->|Clone, set_params, fit, predict| D[BaseModel]
    D -->|Predictions, scores| B
    B -->|best_params, best_score| A
```