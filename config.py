from utils.gep_utils import Bounds

EMMC_B = Bounds()
EMMC_B.add('agent_x', [288.3, 294.7])
EMMC_B.add('agent_z', [433.3, 443.7])
EMMC_B.add('pickaxe_x', [288.3, 294.7])
EMMC_B.add('pickaxe_z', [433.3, 443.7])
EMMC_B.add('shovel_x', [288.3, 294.7])
EMMC_B.add('shovel_z', [433.3, 443.7])
for i in range(5):
    EMMC_B.add('block_' + str(i), [-1, 1])
EMMC_B.add('cart_x', [285, 297])




def get_env_bounds(name):
    if name == 'emmc_env':
        return EMMC_B
    else:
        print('UNKNOWN ENV')