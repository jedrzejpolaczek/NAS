```mermaid
classDiagram
    class ClassificationModel {
        -config: dict
        -logger: Logger
    }
    class RandomForestClassifier {
        -config: dict
        -logger: Logger
        -n_estimators: int
        -max_depth: int
        -min_samples_split: int
        -model: list[dict]
        -classes_: ndarray
        +__init__(config, logger)
        -_build_tree(X, y, depth) dict
        -_to_tensor(data) tf.Tensor
        +fit(X, y)
        -_predict_tree(tree, x) int
        +predict(X) ndarray
        +predict_proba(X) ndarray
        +evaluate(X, y) dict
        +get_params(deep) dict
        +set_params(**params) self
    }
    ClassificationModel <|.. RandomForestClassifier : Inherits
```
```mermaid
sequenceDiagram
    participant C as Client
    participant R as RandomForestClassifier
    participant L as Logger
    C->>R: fit(input_features, target_labels)
    R->>R: _to_tensor(input_features) -> X
    R->>R: _to_tensor(target_labels) -> y
    R->>R: np.unique(y) -> classes_
    loop n_estimators times
        R->>R: np.random.choice(n_samples) -> indices
        R->>R: tf.gather(X, indices) -> X_sample
        R->>R: tf.gather(y, indices) -> y_sample
        R->>R: _build_tree(X_sample, y_sample)
        R->>R: Append tree to model
    end
    R-->>C: (No return, updates self.model)
```
```mermaid
sequenceDiagram
    participant C as Client
    participant R as RandomForestClassifier
    C->>R: predict(input_features)
    R->>R: _to_tensor(input_features) -> X
    loop Over trees in model
        loop Over samples in X
            R->>R: _predict_tree(tree, X[j]) -> prediction
        end
    end
    R->>R: np.apply_along_axis(bincount.argmax) -> final predictions
    R-->>C: Return predictions
```
```mermaid
stateDiagram-v2
    [*] --> Uninitialized
    Uninitialized --> Initialized : __init__(config, logger)
    Initialized --> Fitting : fit(X, y)
    Fitting --> Trained : Trees built, model set
    Trained --> Predicting : predict(X) or predict_proba(X)
    Trained --> Evaluating : evaluate(X, y)
    Predicting --> Trained : Predictions returned
    Evaluating --> Trained : Metrics returned
    Trained --> Initialized : set_params() resets or modifies
    Trained --> [*]
```
```mermaid
graph TD
    A[Root Node] -->|feature_idx=0, threshold=5| B[Left: <= 5]
    A -->|feature_idx=0, threshold=5| C[Right: > 5]
    B -->|feature_idx=1, threshold=10| D[Left: <= 10]
    B -->|feature_idx=1, threshold=10| E[Right: > 10]
    C --> F[Leaf: Class 1]
    D --> G[Leaf: Class 0]
    E --> H[Leaf: Class 1]
```
```mermaid
graph TD
    A[Client] -->|input_features, target_labels| B[RandomForestClassifier]
    B -->|X, y| B : _to_tensor()
    B -->|X_sample, y_sample| B : fit() builds trees
    B -->|input_features| B : predict() or predict_proba()
    B -->|predictions| A
    B -->|input_features, target_labels| B : evaluate()
    B -->|metrics| A
```