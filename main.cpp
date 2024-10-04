#include "kernel.cuh"
#include <iostream>
#include <SFML/Graphics.hpp>
#include <imgui.h>
#include <imgui-SFML.h>

#define WIDTH 800
#define HEIGHT 600
#define FRAME_LIMIT 144
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
    int height = HEIGHT;
    int width = WIDTH;

    if (RENDER)
   
    {
        sf::RenderWindow window(sf::VideoMode(width, height), "CUDA Path Tracer");
        window.setFramerateLimit(FRAME_LIMIT);
        
        ImGui::SFML::Init(window);
        
        sf::Texture texture;
        texture.create(width, height);

        sf::Sprite sprite(texture);
        sf::Clock clock;
        sf::Clock timeClock;

        sf::Uint8* intFrameBuffer = new sf::Uint8[width * height * 4];

        //TODO: maybe refactor so that Z is world up?
        float3 worldUp = make_float3(0.0f, 1.0f, 0.0f);
        float3 cameraPos = make_float3(0.0f, 0.0f, 0.0f);
        float3 cameraDir = make_float3(0.0f, 0.0f, 1.0f);
        float speed = 0.05f;;

        while (window.isOpen())
        {
            // compute FPS
            // start the clock
            sf::Uint64 start = clock.getElapsedTime().asMicroseconds();
            sf::Uint64 timeStart = timeClock.getElapsedTime().asMicroseconds();

            sf::Event event;
            while (window.pollEvent(event))
            {
                ImGui::SFML::ProcessEvent(window, event);
                if (event.type == sf::Event::Closed)
                    window.close();

                //handle resizing without deformation
                // catch the resize events
                if (event.type == sf::Event::Resized)
                {
                    // reallocate a frame buffer
                    width = (int)event.size.width;
                    height = (int)event.size.height;
                    // update the view to the new size of the window
                    sf::FloatRect visibleArea(0, 0, event.size.width, event.size.height);
                    window.setView(sf::View(visibleArea));
                    sf::Uint8* newFB = new sf::Uint8[width*height*4];
                    delete[] intFrameBuffer;
                    intFrameBuffer = newFB;
                

                    texture.create(width,height);
                    sprite = sf::Sprite(texture);
                    std::cout << "current size " << height << " " << width << std::endl;
                }
            }


            sf::Uint32 time = timeClock.getElapsedTime().asMilliseconds();
            float timef = ((float)time) / 1000.0f; //time as float to be used as a kind of uniform
            
            SphereKernelWrapper(intFrameBuffer, width, height, 0.0f, cameraPos, cameraDir);

            sf::Uint64 end = timeClock.getElapsedTime().asMicroseconds();
            
            ImGui::SFML::Update(window, clock.restart());

            ImGui::ShowDemoWindow();

            ImGui::Begin("Hello, world!");
            ImGui::Button("Look at this pretty button");
            ImGui::End();

            texture.update(intFrameBuffer);
            window.draw(sprite);
            ImGui::SFML::Render(window);
            window.display();




            if (VERBOSE)
            {
                printf("\r FPS: %f", 1000000.0f / (end - start));
                fflush(stdout);
                // printf("Frame time (ms): %f\n", (float)(end - start) / 1000.0f);
            }


            //TODO: Move input logic to separate func, maybe make a class for the rendering context.

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
                cameraPos = cameraPos - cross(cameraDir, worldUp) * speed;
			}
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q))
            {
                cameraPos = cameraPos + cross(cameraDir, worldUp) * speed;
            }

            // rotate the camera with the i , j, k, l, u , o keys
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::I))
			{
				cameraDir = rotate(cameraDir, cross(cameraDir, worldUp), -0.01f);
			}
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::K))
            {
                cameraDir = rotate(cameraDir, cross(cameraDir, worldUp), 0.01f);
			}
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::J))
	        {
				cameraDir = rotate(cameraDir, worldUp, 0.01f);
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::L))
            {
                cameraDir = rotate(cameraDir, worldUp, -0.01f);
				
            }


            //std::cout << "Camera position: " << cameraPos.x << " " << cameraPos.y << " " << cameraPos.z << std::endl;
            //std::cout << "Camera direction: " << cameraDir.x << " " << cameraDir.y << " " << cameraDir.z << std::endl;
            //std::cout << "Direction Length: " << length(cameraDir) << std::endl;

            // press Q or Esc to quit
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
				window.close();
        }

        ImGui::SFML::Shutdown();
    }

    /*Mesh mesh;
    loadObj("D:/Programmation/C++/cudaPT/data/objs/bunny.obj", mesh);
    std::cout << "Loaded bunny.obj" << std::endl;
    std::cout << "Number of vertices: " << mesh.vertices.size() << std::endl;
    std::cout << "Number of faces: " << mesh.faces.size() << std::endl;
    std::cout << "AABox min: " << mesh.AABB[0].x << " " << mesh.AABB[0].y << " " << mesh.AABB[0].z << std::endl;
    std::cout << "AABox max: " << mesh.AABB[1].x << " " << mesh.AABB[1].y << " " << mesh.AABB[1].z << std::endl;*/
    
    Matrix3x3 R = makeRotation(make_float3(1, 1, 0));
    float3 t = make_float3(1, 0, 0);
    float3 x = make_float3(1, 0, 0);
    printf("R: \n");
    printf(R);
    printf("t: \n");
	printf(t);
    Matrix4x4 M = makeTransform(R, t);
    printf("M: \n");
    printf(M);
    printf("M * x: \n");
    printf(M * x);
    printf("R * x + t: \n");
    printf(R * x + t);
    printf("R * R^T: \n");
    printf(R * transpose(R));
    printf("R norm: \n");
    printf("%f\n", norm(R));
	return 0;
}