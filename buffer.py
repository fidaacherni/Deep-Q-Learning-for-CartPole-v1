import jax 
import chex
from jax import numpy as jnp

from typing import Tuple, Any, Callable

@chex.dataclass(frozen=True)
class ReplayBufferStorage:
    states: chex.Array  # dtype float32, shape [buffer_size, *state_shape]
    actions: chex.Array  # dtype int32, shape [buffer_size, 1]
    rewards: chex.Array  # dtype float32, shape [buffer_size, 1]
    dones: chex.Array  # dtype bool, shape [buffer_size, 1]
    next_states: chex.Array  # dtype float32, shape [buffer_size, *state_shape]
    cursor: chex.Array  # dtype int32, shape [1]
    full: chex.Array  # dtype bool, shape [1]

Transition = chex.ArrayTree  # Transition is a tuple of (state, action, reward, done, next_state)

@chex.dataclass(frozen=True)
class ReplayBuffer:
  init_buffer: Callable[[Any], ReplayBufferStorage]
  """init_buffer: initializes the replay buffer returning an empty one. It may accept any args"""
  add_transition: Callable[[ReplayBufferStorage, Transition], ReplayBufferStorage]
  """add_transition: 
    adds one transition (state, action, reward, done, next_state) to the replay buffer.
    Transition should be a tuple with arrays, each of the corresponding shape expected
    by the storage without the buffer_size dimension. For example actions should be of shape
    [*action_shape]
  """
  sample_transition: Callable[[chex.PRNGKey, ReplayBufferStorage], Transition]
  """sample_transition:
    samples ONE transition from the replay buffer. the format of transition is:
    a tuple of (state, action, reward, done, next_state).
    This function accepts random key of jax to perform random sampling
  """
  
def init_buffer(buffer_size: int, state_shape: Tuple[int]) -> ReplayBufferStorage:
  """ initializes an empty buffer.

  Args:
    buffer_size: int
    state_shape: Tuple[int]
  Returns:
    ReplayBufferStorage
  """
  return ReplayBufferStorage(
    states=jnp.zeros((buffer_size, *state_shape), dtype=jnp.float32),
    actions=jnp.zeros((buffer_size, 1), dtype=jnp.int32),
    rewards=jnp.zeros((buffer_size, 1), dtype=jnp.float32),
    dones=jnp.zeros((buffer_size, 1), dtype=jnp.bool_),
    next_states=jnp.zeros((buffer_size, *state_shape), dtype=jnp.float32),
    cursor=jnp.array(0), # since the buffer is empty, the cursor points at zero
    full=jnp.array(False)
  )

def add_transition(buffer: ReplayBufferStorage, transition: Transition) -> ReplayBufferStorage:
  
  state, action, reward, done, next_state = transition
  cursor = buffer.cursor
  max_buffer_size = buffer.rewards.shape[0]

  new_states = buffer.states.at[cursor].set(state)
  new_actions = buffer.actions.at[cursor].set(action)
  new_rewards = buffer.rewards.at[cursor].set(reward)
  new_dones = buffer.dones.at[cursor].set(done)
  new_next_states = buffer.next_states.at[cursor].set(next_state)

  new_cursor = (cursor + 1) % max_buffer_size

  new_full = jnp.where(new_cursor == 0, True, buffer.full)

  return buffer.replace(
    states=new_states,
    actions=new_actions,
    rewards=new_rewards,
    dones=new_dones,
    next_states=new_next_states,
    cursor=new_cursor,
    full=new_full
  )

def sample_transition(rng: chex.PRNGKey, buffer: ReplayBufferStorage) -> Transition:

  # Generate a random index within the range of the filled buffer
  # Determine the current size of the buffer to sample from without using conditional statements
  buffer_size = jnp.where(buffer.full, buffer.rewards.shape[0], buffer.cursor)
  # Sample a random index within the range of the filled buffer
  index = jax.random.randint(rng, shape=(), minval=0, maxval=buffer_size)

  # Retrieve the sampled transition immutably
  sampled_state = buffer.states[index]
  sampled_action = buffer.actions[index]
  sampled_reward = buffer.rewards[index]
  sampled_done = buffer.dones[index]
  sampled_next_state = buffer.next_states[index]

  # Return the sampled transition as a tuple
  return sampled_state, sampled_action, sampled_reward, sampled_done, sampled_next_state

FIFOBuffer = ReplayBuffer(
    init_buffer=jax.jit(init_buffer, static_argnums=(0, 1)),
    add_transition=jax.jit(add_transition),
    sample_transition=jax.jit(sample_transition)
)
  