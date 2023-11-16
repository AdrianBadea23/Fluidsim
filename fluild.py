import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher
from tqdm import tqdm
import jaxlib

N_ITERATIONS = 30000
REYNOLDS_NUMBER = 80

N_POINTS_X = 300
N_POINTS_Y = 50

CYLINDER_CENTER_INDEX_X = N_POINTS_X // 5
CYLINDER_CENTER_INDEX_Y = N_POINTS_Y // 2
CYLINDER_RADIUS_INDECES = N_POINTS_Y // 9

MAX_HOR_INFL_VEL = 0.04

VISUALIZE = True
PLOT_EVERY_N_STEPS = 300
SKIP_FIRST_ITER = 0

N_DISCRETE_VELOCITIES = 9
LATTICE_VELO = jnp.array([
    [0, 1, 0, -1, 0, 1, -1, -1, 1,],
    [0, 0, 1, 0, -1, 1, 1, -1, -1,]
]
)

LATTICE_INDICES = jnp.array([0,1,2,3,4,5,6,7,8,])

OPPOSITE_LATTICE_INDCES = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6,])

LATTICE_WEIGHTS = jnp.array([
    4/9,                    #0 velocity
    1/9, 1/9, 1/9, 1/9,     #X axis velo 1 2 3 4
    1/36, 1/36, 1/36, 1/36, # 45 deg velo 5 6 7 8
])

RIGHT_VELO = jnp.array([1,5,8])
UP_VELO = jnp.array([2, 5, 6])
LEFT_VELO = jnp.array([3, 6, 7])
DOWN_VELO = jnp.array([4, 7, 8])
PURE_VERT_VELO = jnp.array([0, 2, 4])
PURE_HORI_VELO = jnp.array([0, 1, 3])

def get_density(discrete_velocities):
    density = jnp.sum(discrete_velocities, axis=-1)
    
    return density


def get_macroscopic_velocities(discrete_velocities, density):
    macroscopic_velocities = jnp.einsum(
        "NMQ,dQ->NMd",
        discrete_velocities,
        LATTICE_VELO,
    ) / density[..., jnp.newaxis]

    return macroscopic_velocities


def get_equilibrium_discrete_velocities(macroscopic_velocities, density):

    projected_discrete_velocities = jnp.einsum(
        "dQ,NMd->NMQ",
        LATTICE_VELO,
        macroscopic_velocities,
    )

    macroscopic_velocities_magnitude = jnp.linalg.norm(
        macroscopic_velocities,
        axis = -1,
        ord = 2,
    )

    get_equilibrium_discrete_velocities = (

        density[..., jnp.newaxis]
        *
        LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, :]
        *
        (1
         +
         3 * projected_discrete_velocities
         +
         9/2 * projected_discrete_velocities**2
         -3/2 * macroscopic_velocities_magnitude[..., jnp.newaxis]**2
        )

    )
    return get_equilibrium_discrete_velocities
    
def main():

    jax.config.update("jax_enable_x64", True)

    kinematic_viscosity = (

        (
            MAX_HOR_INFL_VEL
            *
            CYLINDER_RADIUS_INDECES
        ) / (

            REYNOLDS_NUMBER

        )

    )

    relaxation_omega = (
        (
            1.0
        )/(
            3.0
            *
            kinematic_viscosity
            +
            0.5
        )
    )

    # define a mesh
    x = jnp.arange(N_POINTS_X)
    y = jnp.arange(N_POINTS_Y)
    X, Y = jnp.meshgrid(x,y,indexing="ij")

    # Obstacle mask: array of shape like x or y,  but containsa true if it contains an obj false otherwise

    obstacle_mask = (

        jnp.sqrt(
            (
                X
                -
                CYLINDER_CENTER_INDEX_X
            )**2
            +
            (
                Y
                -
                CYLINDER_CENTER_INDEX_Y
            )**2
        )
        <
            CYLINDER_RADIUS_INDECES

    )

    velocity_profile = jnp.zeros((N_POINTS_X, N_POINTS_Y, 2))
    velocity_profile = velocity_profile.at[:, :, 0].set(MAX_HOR_INFL_VEL)

    @jax.jit
    def update(discrete_velocities_prev):
        # prescribe the outflow boundry cond on the right boundry
        discrete_velocities_prev = discrete_velocities_prev.at[-1, :, LEFT_VELO].set(
            discrete_velocities_prev[-2,:,LEFT_VELO]
        )

        # macroscopic velocities
        density_prev = get_density(discrete_velocities_prev)
        macroscopic_velocities_prev = get_macroscopic_velocities(
            discrete_velocities_prev,
            density_prev,
        )

        #prescribe inflow dirichlet using zou/he scheme

        macroscopic_velocities_prev = macroscopic_velocities_prev.at[0,1:-1,:].set(velocity_profile[0,1:-1,:])
        density_prev = density_prev.at[0, :].set(
            (
                get_density(discrete_velocities_prev[0, :, PURE_VERT_VELO].T)
                +
                2 *
                get_density(discrete_velocities_prev[0, :, LEFT_VELO].T)
            ) / (
                1 - macroscopic_velocities_prev[0, :, 0]
            )
        )

        #compute discrete equil velos

        equilibrium_discrete_velocities = get_equilibrium_discrete_velocities(
            macroscopic_velocities_prev,
            density_prev,
        )

        discrete_velocities_prev = discrete_velocities_prev.at[0,:,RIGHT_VELO].set(equilibrium_discrete_velocities[0,:,RIGHT_VELO])

        discrete_velocities_post_colission = (

            discrete_velocities_prev
            -
            relaxation_omega
            *(
                discrete_velocities_prev
                -
                equilibrium_discrete_velocities
            )

        )

        #bounce back conditions to enforce no-slip

        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_post_colission = discrete_velocities_post_colission.at[obstacle_mask, LATTICE_INDICES[i]].set(discrete_velocities_prev[obstacle_mask,OPPOSITE_LATTICE_INDCES[i]])
    
        #stream alongside lattice velos
        discrete_velocities_streamed = discrete_velocities_post_colission
        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_streamed = discrete_velocities_streamed.at[:,:,i].set(jnp.roll(jnp.roll(discrete_velocities_post_colission[:,:,i],LATTICE_VELO[0,i],axis=0,),LATTICE_VELO[1,i],axis=1))

        return discrete_velocities_streamed
    
    discrete_velocities_prev = get_equilibrium_discrete_velocities(
        velocity_profile,
        jnp.ones((N_POINTS_X,N_POINTS_Y)),
    )

    for iteration_index in tqdm(range(N_ITERATIONS)):
        discrete_velocities_next = update(discrete_velocities_prev)
        discrete_velocities_prev = discrete_velocities_next

        if iteration_index % PLOT_EVERY_N_STEPS == 0 and VISUALIZE and iteration_index > SKIP_FIRST_ITER:
            density = get_density(discrete_velocities_next)
            macroscopic_velocities = get_macroscopic_velocities(
                discrete_velocities_next,
                density,
            )
            velocity_magnitude = jnp.linalg.norm(
                macroscopic_velocities,
                axis = -1,
                ord = 2,
            )

            d_u_d_x, d_u_d_y = jnp.gradient(macroscopic_velocities[...,0])
            d_v_d_x, d_v_d_y = jnp.gradient(macroscopic_velocities[...,1])
            curl = (d_u_d_y - d_u_d_x)

            plt.subplot(211)
            plt.contour(
                X,
                Y,
                velocity_magnitude,
                levels = 50,
                cmap = cmasher.amber,
            )

            plt.colorbar().set_label("Velocity magnitude")
            plt.gca().add_patch(plt.Circle(
                (CYLINDER_CENTER_INDEX_X,CYLINDER_CENTER_INDEX_Y),
                CYLINDER_RADIUS_INDECES,
                color = "red",
            ))

            plt.subplot(212)
            plt.contourf(
                X,
                Y,
                curl,
                levels = 50,
                cmap = cmasher.redshift,
                vmin = -0.02,
                vmax = 0.02,
            )

            plt.colorbar().set_label("Vorticity magnitude")
            plt.gca().add_patch(plt.Circle(
                (CYLINDER_CENTER_INDEX_X,CYLINDER_CENTER_INDEX_Y),
                CYLINDER_RADIUS_INDECES,
                color = "red",
            ))
            plt.draw()
            plt.pause(0.0001)
            plt.clf()



if __name__ == "__main__":
    main()