from generator import Game
import time
import numpy as np
import random
import scipy.misc

tot_frames = 0
while(tot_frames <= 1000):
    yay = Game()
    start_time = time.time()
    count = 0
    for i in range(300):
        if not yay.game_over:
            yay.next_turn()
            # print(yay.turn, " ", yay.player['block_on']['top_left'])#['top_left'])
        count += 1
    end_time = time.time()
    print("Time elapsed: " + str(end_time-start_time))
    if yay.turn > 10:
        tot_frames += yay.turn
    print(len(yay.actions))
    print(yay.turn, " ", tot_frames)