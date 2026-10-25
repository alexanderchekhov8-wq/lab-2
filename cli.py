import click
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from solver import AdvectionSolver, profile_step, profile_gaussian, profile_parabola


# --- Словарь переводов для подписей схем ---
SCHEME_LABELS = {
    "upwind": "Против потока",
    "ftcs": "FTCS",
    "lax": "Лакса",
    "lax_wendroff": "Лакса–Вендроффа",
}


@click.group()
def cli():
    pass


@cli.command()
@click.option("--output", default="stability_test.png")
def test_stability(output: str):
    """Проверка устойчивости схем при Co = 0.8 и Co = 1.2."""
    solver = AdvectionSolver(L=20.0, N=200, u=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    schemes = ["upwind", "ftcs", "lax", "lax_wendroff"]
    titles = [SCHEME_LABELS[s] for s in schemes]

    for idx, (scheme, title) in enumerate(zip(schemes, titles)):
        ax = axes[idx // 2, idx % 2]

        s_stable, _ = solver.solve(scheme, profile_gaussian, Co=0.8, t_final=3.0)
        s_unstable, _ = solver.solve(scheme, profile_gaussian, Co=1.2, t_final=3.0)
        s_exact = solver.exact_solution(profile_gaussian, t=3.0)

        ax.plot(solver.x, s_exact, "k--", label="Точное", alpha=0.7)
        if not np.any(np.isnan(s_stable)) and np.max(np.abs(s_stable)) < 5.0:
            ax.plot(solver.x, s_stable, "b-", label="Co = 0.8 (устойчиво)")
        else:
            ax.text(0.5, 0.5, "Неустойчиво", transform=ax.transAxes, color="red")

        if not np.any(np.isnan(s_unstable)) and np.max(np.abs(s_unstable)) < 5.0:
            ax.plot(solver.x, s_unstable, "r-", label="Co = 1.2")
        else:
            ax.plot(solver.x, np.clip(s_unstable, -2.0, 3.0), "r:",
                    label="Co = 1.2 (неустойчиво)")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x, м")
        ax.set_ylabel("s(x, t)")
        ax.set_ylim(-0.5, 1.8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("Исследование устойчивости схем (критерий КФЛ)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    click.echo(f"Сохранено: {output}")


@cli.command()
@click.option("--output-prefix", default="comparison")
def compare_schemes(output_prefix: str):
    """Сравнение схем на трёх профилях в три момента времени."""
    solver = AdvectionSolver(L=20.0, N=200, u=1.0)

    profiles = {
        "step": (profile_step, "Ступенька"),
        "gaussian": (profile_gaussian, "Гауссов импульс"),
        "parabola": (profile_parabola, "Парабола"),
    }
    times = [2.0, 6.0, 12.0]
    Co = 0.75

    schemes = [
        ("upwind", "Против потока", "tab:blue"),
        ("lax", "Лакса", "tab:orange"),
        ("lax_wendroff", "Лакса–Вендроффа", "tab:green"),
    ]

    for p_name, (p_fn, p_label) in profiles.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for t_idx, t in enumerate(times):
            ax = axes[t_idx]
            s_exact = solver.exact_solution(p_fn, t)
            ax.plot(solver.x, s_exact, "k--", lw=2, label="Точное")

            for s_key, s_title, s_color in schemes:
                s_num, _ = solver.solve(s_key, p_fn, Co=Co, t_final=t)
                ax.plot(solver.x, s_num, color=s_color, label=s_title)

            ax.set_title(f"t = {t} с", fontsize=12)
            ax.set_xlabel("x, м")
            if t_idx == 0:
                ax.set_ylabel("s(x, t)")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(fontsize=8)

        plt.suptitle(f"Сравнение схем (Co = {Co}) — профиль: {p_label}", fontsize=14)
        plt.tight_layout()
        fname = f"{output_prefix}_{p_name}.png"
        plt.savefig(fname, dpi=300)
        click.echo(f"Сохранено: {fname}")


@cli.command()
@click.option("--output", default="courant_effect.png")
def study_courant(output: str):
    """Влияние числа Куранта на схему «против потока»."""
    solver = AdvectionSolver(L=20.0, N=200, u=1.0)
    t_target = 8.0
    co_values = [0.1, 0.5, 0.8, 1.0]

    plt.figure(figsize=(10, 6))
    s_exact = solver.exact_solution(profile_gaussian, t_target)
    plt.plot(solver.x, s_exact, "k--", lw=2.5, label="Точное решение")

    colors = ["purple", "red", "blue", "green"]
    for Co, col in zip(co_values, colors):
        s_num, _ = solver.solve("upwind", profile_gaussian, Co=Co, t_final=t_target)
        l2_err = np.sqrt(np.mean((s_num - s_exact) ** 2))
        plt.plot(solver.x, s_num, color=col, lw=1.8,
                 label=f"Co = {Co} (L2-ошибка = {l2_err:.4f})")

    plt.title("Влияние числа Куранта на схему «против потока» (t = 8.0 с)", fontsize=13)
    plt.xlabel("x, м")
    plt.ylabel("s(x, t)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    click.echo(f"Сохранено: {output}")


@cli.command()
@click.option("--scheme", default="upwind",
              type=click.Choice(["upwind", "lax", "lax_wendroff"]),
              help="Схема для анимации")
@click.option("--profile", "profile_name", default="gaussian",
              type=click.Choice(["step", "gaussian", "parabola"]),
              help="Начальный профиль")
@click.option("--co", default=0.75, type=float, help="Число Куранта")
@click.option("--t-final", default=20.0, type=float, help="Конечное время, с")
@click.option("--fps", default=20, type=int, help="Кадров в секунду")
@click.option("--output", default="animation.gif", help="Файл .gif или .mp4")
def animate(scheme: str, profile_name: str, co: float, t_final: float,
            fps: int, output: str):
    """Анимация переноса профиля (GIF/MP4)."""
    solver = AdvectionSolver(L=20.0, N=200, u=1.0)
    profiles = {
        "step": profile_step,
        "gaussian": profile_gaussian,
        "parabola": profile_parabola,
    }
    p_fn = profiles[profile_name]

    dt = co * solver.dx / solver.u
    num_steps = int(np.round(t_final / dt))
    stride = max(1, num_steps // 200)

    s = p_fn(solver.x).copy()
    frames = [(0.0, s.copy(), solver.exact_solution(p_fn, 0.0))]
    step_fn = {
        "upwind": solver.step_upwind,
        "lax": solver.step_lax,
        "lax_wendroff": solver.step_lax_wendroff,
    }[scheme]

    t = 0.0
    for n in range(1, num_steps + 1):
        s = step_fn(s, co)
        t = n * dt
        if n % stride == 0 or n == num_steps:
            frames.append((t, s.copy(), solver.exact_solution(p_fn, t)))

    click.echo(f"Кадров: {len(frames)} (шаг сохрания: {stride}, dt = {dt:.4f})")

    fig, ax = plt.subplots(figsize=(9, 5))
    line_num, = ax.plot([], [], "b-", lw=2, label=SCHEME_LABELS[scheme])
    line_exact, = ax.plot([], [], "k--", lw=1.5, alpha=0.7, label="Точное")
    ax.set_xlim(0, solver.L)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("x, м")
    ax.set_ylabel("s(x, t)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    title = ax.set_title("")

    def update(frame_idx):
        t, s_num, s_ex = frames[frame_idx]
        line_num.set_data(solver.x, s_num)
        line_exact.set_data(solver.x, s_ex)
        title.set_text(f"{SCHEME_LABELS[scheme]}, Co = {co}, t = {t:.2f} с")
        return line_num, line_exact, title

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=True
    )

    if output.endswith(".gif"):
        ani.save(output, writer="pillow", fps=fps)
    elif output.endswith(".mp4"):
        ani.save(output, writer="ffmpeg", fps=fps)
    else:
        click.echo("Ошибка: расширение файла должно быть .gif или .mp4")
        return
    plt.close(fig)
    click.echo(f"Сохранено: {output}")


if __name__ == "__main__":
    cli()