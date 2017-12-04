import multiprocessing as mp
import time
import numpy as np
import random
import scipy.misc

class Game(object):

    def __init__(self, size=50, block_height=5, block_width=5, speed=1,
            player_size=2, header_height=None, move_speed=1):

        if type(size) == int:
            assert(size > 0), "Size must be greater than 0"
            size_x = size
            size_y = size
        elif isinstance(size, tuple):
            size_x = size[0]
            size_y = size[1]
            assert(size_x > 0), "Size must be greater than 0"
            assert(size_y > 0), "Size must be greater than 0"
        else:
            assert(False), "Size is the wrong type"


        # Output
        self.boards = []
        self.actions = []
        self.turn = 0
        self.game_over = False
        self.move_speed = move_speed

        # Create Board parameters
        self.board = np.empty((size_x,size_y))
        self.size_y = size_x
        self.size_x = size_y

        #Parameters for below
        num_rows = size_y//(block_height*2)
        block_width = block_width
        num_block_in_row = self.size_x//(block_width*2)



        # Create initial boxes
        # Boxes is just the structure storing all ice blocks and headers
        self.boxes = []
        self.block_height = block_height
        self.block_width = block_width

        #create header
        header = {}
        header['top_left'] = (0,0)
        header['color'] = random.randrange(0,255)
        if header_height is None:
            header_size_y = block_height
        else:
            head_size_y = header_height
        header['size'] = (self.size_x, header_size_y)
        header['speed'] = 0

        self.boxes.append(header)

        # Create blocks
        # There is ablock every other row
        for row in range(2, self.size_y//block_height, 2):
            y = row*block_height
            for x in range(0, block_width*num_block_in_row*2,
                    block_width*2):
                box = {}

                if row % 4 == 0:
                    box['speed'] = speed
                else:
                    box['speed'] = -speed

                box['color'] = random.randrange(0,255)
                box['top_left'] = (x, y)
                box['size'] = (block_width, block_height)
                self.boxes.append(box)

        #Player
        self.player = {}
        self.player['top_left'] = (0,0)
        self.player['color'] = random.randrange(0,255)
        self.player['size'] = (3, 3)
        self.player['block_on'] = header

        self.render()

    def render(self):
        assert(self.boxes is not None)
        for x in range(self.size_x):
            for y in range(self.size_y):
                self.board[y][x] = random.randrange(0,255)

        for box in self.boxes:
            (x_min, y_min) = box['top_left']
            (size_x, size_y) = box['size']
            for x in range(x_min, x_min + size_x):
                for y in range(y_min, y_min + size_y):
                    self.set_color(x, y, box['color'])

        (x_min, y_min) = self.player['top_left']
        (size_x, size_y) = self.player['size']
        for x in range(x_min, x_min+size_x):
            for y in range(y_min, y_min+size_y):
                self.set_color(x, y, self.player['color'])

        self.boards.append(self.board)
        self.turn += 1
        scipy.misc.toimage(self.board).save('output' + str(self.turn) + '.png')

    def next_turn(self):
        self.update_locations()
        self.make_move()
        self.render()

    def make_move(self):
        if self.game_over:
            return

        (player_x_min, player_y_min) = self.player['top_left']
        (player_size_x, player_size_y) = self.player['size']
        (player_x_max, player_x_max) = (player_x_min + player_size_x,
                player_y_min + player_size_y)

        block = self.player['block_on']
        (box_x_min, box_y_min) = block['top_left']
        (box_size_x, box_size_y) = block['size']
        (box_x_max, box_x_max) = (box_x_min + box_size_x,
                box_y_min + box_size_y)

        # Get all possible moves
        possible_moves = []
        # 0 is right
        # 1 is down
        # 2 is left
        # 3 is up
        # 4 skip block right
        # 5 skip block down
        # 6 skip block left
        # 7 skip block up

        if(player_x_max + self.move_speed <  box_x_max):
            possible_moves.append(0)

        if(player_y_max + self.move_speed <  box_y_max):
            possible_moves.append(1)

        if(player_x_min - self.move_speed <  box_x_min):
            possible_moves.append(2)


        if(player_y_min - self.move_speed <  boy_y_min):
            possible_moves.append(3)
        pass

    def set_color(self, x, y, color):
        if (x < 0 or x >= self.size_x):
            return
        if (y < 0 or y >= self.size_y):
            return
        self.board[y][x] = color

    def update_locations(self):
        new_boxes = []
        (player_x_min, player_y_min) = self.player['top_left']

        # Update block locations
        for box in self.boxes:
            (x_min, y_min) = box['top_left']

            speed = box['speed']
            (size_x, size_y) = box['size']
            x_max = x_min + size_x
            y_max = y_min + size_y


            x_min += speed #Update x_min

            new_box = box.copy()

            #Check if we are on screen and gerenate new_block
            if x_min >= self.size_x:
                new_box['color'] = random.randrange(0,255)
                x_min = 0
            elif x_max < 0:
                new_box['color'] = random.randrange(0,255)
                x_min = self.size_x - 1 - size_x

            # Add block to new_list
            new_box['top_left'] = (x_min, y_min)
            new_boxes.append(new_box)

        #update player location
        block = self.player['block_on']
        speed = block['speed']
        self.player['top_left'] = (player_x_min+speed, player_y_min)


        (player_x_min, player_y_min) = self.player['top_left']
        (player_size_x, player_size_y) = self.player['size']
        player_x_max = player_x_min + player_size_x
        if player_x_min < 0 or  player_x_max >= self.size_x:
            self.game_over = True

        self.boxes = new_boxes

        self.render()

    def get_outputs(self):
        return zip(self.boards, self.actions)

    def run_until_over(self):
        while not self.game_over:
            self.next_turn()
        print(self.turn)


yay = Game()
start_time = time.time()
yay.run_until_over()
end_time = time.time()
print("Time elapsed: " + str(end_time-start_time))
