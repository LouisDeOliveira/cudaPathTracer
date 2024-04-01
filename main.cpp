#include "kernel.cuh"
#include <iostream>
#include <SFML/Graphics.hpp>

#define WIDTH 800
#define HEIGHT 600
#define FRAME_LIMIT 60
#define RENDER 1


int main(int argc, char **argv) {
	// Initialize the kernel

	std::cout << "Hello, World!" << std::endl;

	//debugKernelWrapper();

    if (RENDER)
   
    {
        sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "SFML works!");
        window.setFramerateLimit(FRAME_LIMIT);
        sf::Texture texture;
        texture.create(WIDTH, HEIGHT);

        sf::Sprite sprite(texture);
        sf::Clock clock;

        sf::Uint8* intFrameBuffer = new sf::Uint8[WIDTH * HEIGHT * 4];
        float3 cameraPos = make_float3(0.0f, 0.0f, 0.0f);
        float speed = 0.05f;;

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
            renderKernelWrapper(intFrameBuffer, WIDTH, HEIGHT, timef, cameraPos);

            texture.update(intFrameBuffer);
            window.draw(sprite);
            window.display();

            sf::Uint64 end = clock.getElapsedTime().asMicroseconds();

            printf("FPS: %f\n", 1000000.0f / (end - start));
            printf("Frame time (ms): %f\n", (float)(end - start) / 1000.0f);

            // move the camera with the arrow keys
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
			{
				cameraPos.x -= speed;
			}
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
               {
                cameraPos.x += speed;
			}
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))
            {
                cameraPos.z -= speed;
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
			{
				cameraPos.z += speed;
			}
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
			{
				cameraPos.y += speed;
			}
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
			{
				cameraPos.y -= speed;
			}
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