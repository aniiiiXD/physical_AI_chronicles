import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Training Data
# -------------------------

X = np.array([1, 2, 3, 4], dtype=float)
Y_true = np.array([5, 10, 15, 20], dtype=float)

# -------------------------
# ReLU
# -------------------------

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

# -------------------------
# Random Initialization
# -------------------------

w = np.random.randn()
b = np.random.randn()

learning_rate = 0.1

# -------------------------
# Live Plot Setup
# -------------------------

plt.ion()
fig = plt.figure(figsize=(14, 7))
fig.canvas.manager.set_window_title("Neural Net Training — LIVE")

gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.35, wspace=0.3)

ax_fit = fig.add_subplot(gs[:, 0])
ax_loss = fig.add_subplot(gs[0, 1])
ax_params = fig.add_subplot(gs[1, 1])

x_line = np.linspace(0, 5, 200)

loss_history = []
w_history = []
b_history = []

ax_fit.set_title("Model Fit (updates every epoch)")
ax_fit.set_xlabel("x")
ax_fit.set_ylabel("y")
ax_fit.set_xlim(0, 5)
ax_fit.set_ylim(-2, 25)
ax_fit.grid(True, alpha=0.3)

ax_loss.set_title("Loss")
ax_loss.set_xlabel("epoch")
ax_loss.set_ylabel("MSE")
ax_loss.grid(True, alpha=0.3)

ax_params.set_title("Weight & Bias")
ax_params.set_xlabel("epoch")
ax_params.grid(True, alpha=0.3)

true_scatter = ax_fit.scatter(X, Y_true, s=120, c="#00ff88", edgecolors="white", linewidths=1.5, label="Y_true", zorder=5)
pred_scatter = ax_fit.scatter(X, np.zeros_like(X), s=120, c="#ff3366", edgecolors="white", linewidths=1.5, label="Y_pred", zorder=5)
fit_line, = ax_fit.plot([], [], color="#ffaa00", linewidth=2.5, label="relu(w·x + b)")
residual_lines = [ax_fit.plot([], [], color="#ff3366", alpha=0.35, linewidth=1)[0] for _ in X]
ax_fit.legend(loc="upper left")

loss_line, = ax_loss.plot([], [], color="#00ccff", linewidth=2)
w_line, = ax_params.plot([], [], color="#ffaa00", linewidth=2, label="w")
b_line, = ax_params.plot([], [], color="#cc66ff", linewidth=2, label="b")
ax_params.legend(loc="upper right")

status_text = fig.text(0.02, 0.97, "", fontsize=11, family="monospace", color="#cccccc")

for epoch in range(100):

    total_loss = 0

    grad_w = 0
    grad_b = 0

    # Loop over all training samples

    for x, yt in zip(X, Y_true):

        # -------------------------
        # Forward Pass
        # -------------------------

        z = w * x + b

        y = relu(z)

        # -------------------------
        # Loss
        # -------------------------

        loss = (yt - y) ** 2

        total_loss += loss

        # -------------------------
        # Backpropagation
        # -------------------------

        dL_dy = 2 * (y - yt)

        dy_dz = relu_derivative(z)

        dz_dw = x

        dz_db = 1

        grad_w += dL_dy * dy_dz * dz_dw
        grad_b += dL_dy * dy_dz * dz_db

    # Average loss over dataset

    total_loss /= len(X)

    grad_w /= len(X)
    grad_b /= len(X)

    # -------------------------
    # Gradient Descent
    # -------------------------

    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    # -------------------------
    # Update Live Plot
    # -------------------------

    y_pred = relu(w * X + b)
    y_curve = relu(w * x_line + b)

    loss_history.append(total_loss)
    w_history.append(w)
    b_history.append(b)

    epochs_so_far = np.arange(len(loss_history))

    fit_line.set_data(x_line, y_curve)
    pred_scatter.set_offsets(np.column_stack([X, y_pred]))

    for i, (yt, yp) in enumerate(zip(Y_true, y_pred)):
        residual_lines[i].set_data([X[i], X[i]], [yt, yp])

    loss_line.set_data(epochs_so_far, loss_history)
    ax_loss.relim()
    ax_loss.autoscale_view()

    w_line.set_data(epochs_so_far, w_history)
    b_line.set_data(epochs_so_far, b_history)
    ax_params.relim()
    ax_params.autoscale_view()

    status_text.set_text(
        f"epoch {epoch:3d}/99  |  loss={total_loss:.4f}  |  w={w:+.4f}  |  b={b:+.4f}"
    )

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.03)

    if epoch % 10 == 0:
        print(
            f"Epoch={epoch:5d} "
            f"Loss={total_loss:.6f} "
            f"w={w:.4f} "
            f"b={b:.4f}"
        )

plt.ioff()
print("\nTraining Complete")
print(f"Final Weight = {w:.4f}")
print(f"Final Bias   = {b:.4f}")

print("\nFinal Predictions vs Y_true")
for x, yt in zip(X, Y_true):
    y_pred = relu(w * x + b)
    print(f"x={x:.0f}  Y_true={yt:.0f}  Y_pred={y_pred:.4f}")

plt.show()
