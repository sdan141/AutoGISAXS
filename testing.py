import numpy as np

# Example data
X = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
y = np.array([0, 1, 2, 3, 4])

# Function to create custom train-test splits with a maximum number of splits
def custom_train_test_split(X, y, max_splits=10, test_size=0.2):
    num_samples = len(X)
    num_test_samples = int(num_samples * test_size)
    if num_test_samples == 0:
        num_test_samples = 1  # Ensure at least one sample in the test set

    indices = np.arange(num_samples)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)

    splits = []
    for i in range(max_splits):
        if i * num_test_samples >= num_samples:
            break

        test_indices = indices[i * num_test_samples: (i + 1) * num_test_samples]
        train_indices = np.setdiff1d(indices, test_indices)

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        splits.append((X_train, X_test, y_train, y_test))

    return splits

# Create custom splits
splits = custom_train_test_split(X, y, max_splits=2, test_size=0.2)

# Print results
for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
    print(f"Split {i+1}:")
    print("X_train:", X_train)
    print("y_train:", y_train)
    print("X_test:", X_test)
    print("y_test:", y_test)
    print()
