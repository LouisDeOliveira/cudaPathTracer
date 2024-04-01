#include "kernel.cuh"
#include <iostream>
#include <SFML/Graphics.hpp>

#define WIDTH 800
#define HEIGHT 600
#define FRAME_LIMIT 60
#define RENDER 1
#define VERBOSE 0

float3 rotate(const float3 v, const float3 axis, float angle)
{
	float3 res;
	float c = cos(angle);
	float s = sin(angle);
	float t = 1 - c;
	float x = axis.x;
	float y = axis.y;
	float z = axis.z;

	res.x = (t * x * x + c) * v.x + (t * x * y - s * z) * v.y + (t * x * z + s * y) * v.z;
	res.y = (t * x * y + s * z) * v.x + (t * y * y + c) * v.y + (t * y * z - s * x) * v.z;
	res.z = (t * x * z - s * y) * v.x + (t * y * z + s * x) * v.y + (t * z * z + c) * v.z;

	return res;
}

void saveAsPNG(const char* filename, const sf::Uint8* pixels, unsigned int width, unsigned int height)
{
    sf::Image image;
    image.create(width, height, pixels);
    image.saveToFile(filename);
}

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
        float3 cameraDir = make_float3(0.0f, 0.0f, 1.0f);
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
            SphereKernelWrapper(intFrameBuffer, WIDTH, HEIGHT, timef, cameraPos, cameraDir);

            texture.update(intFrameBuffer);
            window.draw(sprite);
            window.display();

            sf::Uint64 end = clock.getElapsedTime().asMicroseconds();

            if (VERBOSE)
            {
                printf("FPS: %f\n", 1000000.0f / (end - start));
                printf("Frame time (ms): %f\n", (float)(end - start) / 1000.0f);
            }

            // move the camera forward and backward with z and s
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Z))
			{
				cameraPos = cameraPos - cameraDir * speed;
			}
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
			{
				cameraPos = cameraPos + cameraDir * speed;
           			}
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
            {
                cameraPos = cameraPos + cross(cameraDir, make_float3(0.0f, 1.0f, 0.0f)) * speed;
			}
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q))
            {
                cameraPos = cameraPos - cross(cameraDir, make_float3(0.0f, 1.0f, 0.0f)) * speed;
            }
            // rotate the camera with i j k 

            //std::cout << "Camera position: " << cameraPos.x << " " << cameraPos.y << " " << cameraPos.z << std::endl;
            //std::cout << "Camera direction: " << cameraDir.x << " " << cameraDir.y << " " << cameraDir.z << std::endl;
            //std::cout << "Direction Length: " << length(cameraDir) << std::endl;

            // press Q or Esc to quit
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
				window.close();
        }
    }

    //Mesh mesh;
    //loadObj("D:/Programmation/C++/cudaPT/data/objs/bunny.obj", mesh);
    //std::cout << "Loaded bunny.obj" << std::endl;
    //std::cout << "Number of vertices: " << mesh.vertices.size() << std::endl;
    //std::cout << "Number of faces: " << mesh.faces.size() << std::endl;
    //std::cout << "AABox min: " << mesh.AABB[0].x << " " << mesh.AABB[0].y << " " << mesh.AABB[0].z << std::endl;
    //std::cout << "AABox max: " << mesh.AABB[1].x << " " << mesh.AABB[1].y << " " << mesh.AABB[1].z << std::endl;
    
	return 0;
}