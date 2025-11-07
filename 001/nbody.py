import torch


def nbody_step(positions, velocities):
    timestep = 0.01
    # get the distance between the particles
    # first shape expansion from N, D -> N, 1, D and N, D -> 1, N, D
    # then broadcasting subtraction from N, 1, D - 1, N, D -> N, N, D
    difference = positions[None, :, :] - positions[:, None, :]
    # then norm calculation from N, N, D -> N, N
    distances = torch.norm(difference, dim=2)
    # get the force between the particles
    forces = 1 / distances**3
    # mask the distance to itself
    forces = torch.where(distances == 0, 0, forces)
    # get the acceleration of the particles for each pairwise interaction
    # force unsqueeze from N, N -> N, N, 1
    # then broadcasting multiplication from N, N, 1 * N, N, D -> N, N, D
    accelerations_pairwise = forces.unsqueeze(2) * difference
    # sum the accelerations for each particle
    accelerations = torch.sum(accelerations_pairwise, dim=1)
    # get the new velocity of the particles --> EULER METHOD
    new_velocities = velocities + accelerations * timestep
    # get the new position of the particles
    new_positions = positions + velocities * timestep
    return new_positions, new_velocities


positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

velocities = torch.tensor([[0.0, 0.1], [0.0, 0.0], [0.1, 0.0]])

new_pos, new_vel = nbody_step(positions, velocities)

print("Positions:")
print(new_pos)
print("Velocities:")
print(new_vel)
