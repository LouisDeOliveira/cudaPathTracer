#include "kernel.cuh"
#include <iostream>
#include <SFML/Graphics.hpp>

#define WIDTH 800
#define HEIGHT 800

#define RENDER 0


int main(int argc, char **argv) {
	// Initialize the kernel

	std::cout << "Hello, World!" << std::endl;

	//debugKernelWrapper();

    if (RENDER)
   
    {
        sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "SFML works!");
        sf::Texture texture;
        texture.create(WIDTH, HEIGHT);

        sf::Sprite sprite(texture);
        sf::Clock clock;

        sf::Uint8* intFrameBuffer = new sf::Uint8[WIDTH * HEIGHT * 4];

        while (window.isOpen())
        {
            // compute FPS
            // start the clock
            sf::Uint64 start = clock.getElapsedTime().asMicroseconds();

            sf::Event event;
            while (window.pollEvent(event))
            {
                if (event.type == sf::Event::Closed)
                    window.close();
            }

            sf::Uint32 time = clock.getElapsedTime().asMilliseconds();
            float timef = ((float)time) / 1000.0f;
            uvKernelWrapper(intFrameBuffer, WIDTH, HEIGHT, timef);

            texture.update(intFrameBuffer);
            window.draw(sprite);
            window.display();

            sf::Uint64 end = clock.getElapsedTime().asMicroseconds();

            printf("FPS: %f\n", 1000000.0f / (end - start));
            printf("Frame time (ms): %f\n", (float)(end - start) / 1000.0f);
        }
    }

    Mesh mesh;
    loadObj("D:/Programmation/C++/cudaPT/data/objs/bunny.obj", mesh);
    std::cout << "Loaded bunny.obj" << std::endl;
    std::cout << "Number of vertices: " << mesh.vertices.size() << std::endl;
    std::cout << "Number of faces: " << mesh.faces.size() << std::endl;
    std::cout << "AABox min: " << mesh.AABB[0].x << " " << mesh.AABB[0].y << " " << mesh.AABB[0].z << std::endl;
    std::cout << "AABox max: " << mesh.AABB[1].x << " " << mesh.AABB[1].y << " " << mesh.AABB[1].z << std::endl;
    
	return 0;
}