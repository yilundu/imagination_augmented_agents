from generator import Game
import time
import numpy as np
import random
import scipy.misc

tot_frames = 0
data = []
target = []
start_time = time.time()

data_length = 100000

count_vert = 0
count_horiz = 0

while(len(data) <= data_length):
    yay = Game(player_size=3, toPNG=False)

    count = 0
    
    for i in range(300):
        if not yay.game_over:
            yay.next_turn()
            # print(yay.turn, " ", yay.player['block_on']['top_left'])#['top_left'])
        count += 1
    assert len(yay.actions) + 1 == len(yay.boards)

    size = yay.size_x
    if yay.turn > 10:
        tot_frames += yay.turn - 2
        for i in range(3, len(yay.boards)):
            # one_hot = [np.zeros((size, size)) for j in range(5)]
            # one_hot[yay.actions[i - 1] - 1] = np.tile(1.0, (size, size))
            # one_hot = np.array(one_hot)
            past_frames = np.array([yay.boards[i - 3], yay.boards[i - 2], yay.boards[i - 1]])
            total_input = np.vstack([past_frames, np.tile(yay.actions[i - 1], (1, size, size))])
            curr_frame = np.array(yay.boards[i])
            # print(curr_frame[8:])
            
            if (yay.actions[i - 1]) % 2 == 0 and yay.actions[i - 1] != 4: # vertical move
                # print(yay.actions[i - 1])
                count_vert += 1
                data.append(total_input)
                target.append(curr_frame)
            elif count_horiz < count_vert / 5:#count_horiz < count_vert + 10:
                # print(yay.actions[i - 1])
                count_horiz += 1
                data.append(total_input)
                target.append(curr_frame)
            print(count_horiz, " ", count_vert)
        # last frame, an all-black death screen.
        past_frames = np.array([yay.boards[-3], yay.boards[-2], yay.boards[-1]])
        total_input = np.vstack([past_frames, np.tile(4, (1, size, size))])
        data.append(total_input)
        curr_frame = np.array(np.tile(0, (size, size)))
        target.append(curr_frame)
    print(yay.turn, " ", len(data))

data = (np.array(data)[:data_length]).astype(np.uint8)
target = (np.array(target)[:data_length]).astype(np.uint8)
print(data.shape)
print(target.shape)

np.savez('pretrain_data.npz', data, target)

end_time = time.time()
print("Time elapsed: " + str(end_time-start_time))