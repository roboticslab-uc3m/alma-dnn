import pygame

import sys

DATASET_DIR = "dataset"

IMAGE_SIZE = (100, 100)

if len(sys.argv) < 5:
    print("Usage:  folding_vis.py image_name pick_x pick_y place_x place_y")
    print("Example:  folding_vis.py dataset/image1.png 1 2 3 4")
    quit(1)

image_name = sys.argv[-5]
pick_x = sys.argv[-4]
pick_y = sys.argv[-3]
place_x = sys.argv[-2]
place_y = sys.argv[-1]

pygame.init()
 
scrn = pygame.display.set_mode(IMAGE_SIZE)
 
imp = pygame.image.load(image_name).convert()
 
scrn.blit(imp, (0, 0))
 
pygame.display.flip()
status = True
while (status):
    for i in pygame.event.get():
        if i.type == pygame.QUIT:
            status = False
 
pygame.quit()
