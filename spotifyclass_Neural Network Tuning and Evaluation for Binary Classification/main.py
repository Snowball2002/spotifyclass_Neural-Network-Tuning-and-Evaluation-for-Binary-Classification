# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

data = pd.read_csv('spotify_preprocessed.csv')


data = data.sample(frac=1).reset_index(drop=True)


X = data.drop('target', axis=1)
y = data['target']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


sgd = SGD(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16)


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


def build_and_compile_model(layers, units, learning_rate):
    model = Sequential()
    model.add(Dense(units, input_dim=X_train.shape[1], activation='relu'))
    for _ in range(layers - 1):
        model.add(Dense(units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    sgd = SGD(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


configurations = [
    {'layers': 2, 'units': 32, 'learning_rate': 0.01},
    {'layers': 3, 'units': 64, 'learning_rate': 0.01},
    {'layers': 2, 'units': 128, 'learning_rate': 0.001},

]

results = []

for config in configurations:
    model = build_and_compile_model(config['layers'], config['units'], config['learning_rate'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, verbose=0)
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    results.append((config, train_loss, train_accuracy, val_loss, val_accuracy))


best_config = max(results, key=lambda x: x[4])

print(f"Best Configuration: {best_config[0]}")
print(f"Training Loss: {best_config[1]:.4f}, Training Accuracy: {best_config[2]:.4f}")
print(f"Validation Loss: {best_config[3]:.4f}, Validation Accuracy: {best_config[4]:.4f}")


best_model = build_and_compile_model(best_config[0]['layers'], best_config[0]['units'], best_config[0]['learning_rate'])
best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, verbose=0)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
