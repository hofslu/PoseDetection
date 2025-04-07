import os
os.environ['SDL_VIDEODRIVER'] = 'cocoa'

import pygame

def main():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Test Window")

    running = True
    while running:
        screen.fill((30, 30, 30))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

if __name__ == "__main__":
    main()
