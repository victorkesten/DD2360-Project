#version 330 core
out vec4 FragColor;
//uniform vec4 MVP;
uniform vec4 color;
void main()
{
   FragColor = vec4(1.0f, 0.1f, 0.5f, 0.4f);
   //FragColor = color;
  // FragColor = MVP;
}
