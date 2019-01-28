from utils.gep_utils import Bounds

#EMMC CONFIG
EMMC_B = Bounds()
EMMC_B.add('agent_x', [288.3, 294.7])
EMMC_B.add('agent_z', [432.3, 443.7])
EMMC_B.add('pickaxe_x', [288.3, 294.7])
EMMC_B.add('pickaxe_z', [433.3, 443.7])
EMMC_B.add('shovel_x', [288.3, 294.7])
EMMC_B.add('shovel_z', [433.3, 443.7])
for i in range(5):
    EMMC_B.add('block_' + str(i), [-1, 1])
EMMC_B.add('cart_x', [285, 297])
EMMC_B_motor_states = ['agent_x', 'agent_z']



#ARM CONFIG
ARM_B = Bounds()

ARM_B.add('hand_x', [-1.,1.])
ARM_B.add('hand_y', [-1.,1.])
ARM_B.add('gripper', [-1.,1.])
ARM_B.add('stick1_x', [-1.5,1.5])
ARM_B.add('stick1_y', [-1.5,1.5])
ARM_B.add('stick2_x', [-1.5,1.5])
ARM_B.add('stick2_y', [-1.5,1.5])
ARM_B.add('magnet1_x', [-1.5,1.5])
ARM_B.add('magnet1_y', [-1.5,1.5])
ARM_B.add('magnet2_x', [-1.5,1.5])
ARM_B.add('magnet2_y', [-1.5,1.5])
ARM_B.add('magnet3_x', [-1.5,1.5])
ARM_B.add('magnet3_y', [-1.5,1.5])
ARM_B.add('scratch1_x',[-1.5,1.5])
ARM_B.add('scratch1_y',[-1.5,1.5])
ARM_B.add('scratch2_x',[-1.5,1.5])
ARM_B.add('scratch2_y',[-1.5,1.5])
ARM_B.add('scratch3_x',[-1.5,1.5])
ARM_B.add('scratch3_y',[-1.5,1.5])
ARM_B.add('cat_x',[-1.5,1.5])
ARM_B.add('cat_y',[-1.5,1.5])
ARM_B.add('dog_x',[-1.5,1.5])
ARM_B.add('dog_y',[-1.5,1.5])
ARM_B.add('static1_x',[-1.5,1.5])
ARM_B.add('static1_y',[-1.5,1.5])
ARM_B.add('static2_x',[-1.5,1.5])
ARM_B.add('static2_y',[-1.5,1.5])
ARM_B.add('static3_x',[-1.5,1.5])
ARM_B.add('static3_y',[-1.5,1.5])
ARM_B.add('static4_x',[-1.5,1.5])
ARM_B.add('static4_y',[-1.5,1.5])
ARM_B_motor_states = ['hand_x', 'hand_y','gripper']



def get_env_bounds(name):
    if name == 'emmc_env':
        return EMMC_B
    elif name == 'arm_env':
        return ARM_B
    else:
        print('UNKNOWN ENV')

def get_motor_states(name):
    if name == 'emmc_env':
        return EMMC_B_motor_states
    elif name == 'arm_env':
        return ARM_B_motor_states
    else:
        print('UNKNOWN ENV')

