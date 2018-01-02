#version 330 core
layout (location = 0) in vec3 aPos;
//uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 offset;

void main()
{
  //gl_Position = projection * view * model  * vec4(aPos.x, aPos.y, aPos.z, 1.0) ;
  gl_Position = projection * view * (vec4(aPos*376780.0f+offset, 1.0)) ;//multiply vertex x, y and z by particle diameter, add offset

  //old
   //gl_Position = projection * view * model  * transform*vec4(aPos.x, aPos.y, aPos.z, 1.0) ;

  //  gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0) ;

}
