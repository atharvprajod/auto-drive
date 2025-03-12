#version 450

layout(location = 0) out float fragDepth;

void main() {
    // Only write depth
    fragDepth = gl_FragCoord.z;
} 