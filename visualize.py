import imageio
import numpy as np
import pickle
import gymnasium as gym
import gymnasium_maze


def process_state(observation):
    agent_pos = observation['agent'] / 9
    target_pos = observation['target'] / 9
    visible_holes = observation['visible_holes'].flatten() / 9
    return np.concatenate([agent_pos, target_pos, visible_holes]).astype(np.float32)


with open("agent.pkl", "rb") as fp:
    agent = pickle.load(fp)


def show_video_of_model(agent, env_name, env_size=10):
    # Создаем среду с режимом рендеринга rgb_array
    env = gym.make(env_name, size=env_size, render_mode="rgb_array")

    observation, _ = env.reset()
    state = process_state(observation)
    done = False
    frames = []

    while not done:
        # Получаем кадр и проверяем его размерность
        frame = env.render()

        # Проверяем и преобразуем кадр при необходимости
        if isinstance(frame, np.ndarray):
            # Если кадр уже в формате numpy array
            if frame.ndim == 3 and frame.shape[2] == 3:  # Проверяем формат (H,W,3)
                frames.append(frame)
            else:
                # Пробуем преобразовать в правильный формат
                try:
                    frame = np.asarray(frame)
                    if frame.ndim == 2:
                        # Если 2D, преобразуем в 3D (grayscale -> RGB)
                        frame = np.stack([frame] * 3, axis=-1)
                    frames.append(frame)
                except Exception as e:
                    print(f"Ошибка преобразования кадра: {e}")
                    break
        else:
            print("Неподдерживаемый формат кадра")
            break

        # Получаем действие от агента
        action = agent.act(state)

        # Выполняем шаг в среде
        next_observation, reward, done, _, _ = env.step(action.item())
        next_state = process_state(next_observation)
        state = next_state

        # Ограничиваем максимальную длину видео
        if len(frames) > 80:
            done = True

    env.close()

    if frames:
        try:
            # Сохраняем видео, используя правильный FPS
            imageio.mimsave('gridworld_agent.mp4', frames, fps=4)  # Используем 4 FPS как в метаданных
            print("Видео успешно сохранено!")
        except Exception as e:
            print(f"Ошибка при сохранении видео: {e}")
    else:
        print("Нет кадров для сохранения")


# Запускаем создание видео
show_video_of_model(agent, 'gymnasium_maze/GridWorld-v0', env_size=10)
