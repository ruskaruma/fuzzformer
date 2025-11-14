#include "fuzzformer/visualRenderer.h"

#include <iostream>
#include <vector>

#ifdef FUZZFORMER_HAS_TORCH
#include <torch/torch.h>
#endif

#ifdef FUZZFORMER_HAS_OPENGL
#include <GLFW/glfw3.h>
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#include <cmath>
#endif

namespace fuzzformer {

VisualRenderer::VisualRenderer() : initialized_(false), opengl_initialized_(false), opengl_context_(nullptr) {}

VisualRenderer::~VisualRenderer() {
  shutdown();
  shutdown_opengl();
}

void VisualRenderer::initialize() {
  initialized_ = true;
}

void VisualRenderer::shutdown() {
  initialized_ = false;
}

#ifdef FUZZFORMER_HAS_OPENGL
struct OpenGLContext {
  GLFWwindow* window;
  int width;
  int height;
  std::vector<float> attention_data;
  int attention_width;
  int attention_height;
};

static void glfw_error_callback(int error, const char* description) {
  std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
  glViewport(0, 0, width, height);
  OpenGLContext* ctx = static_cast<OpenGLContext*>(glfwGetWindowUserPointer(window));
  if (ctx) {
    ctx->width = width;
    ctx->height = height;
  }
}
#endif

bool VisualRenderer::initialize_opengl(int width, int height) {
#ifdef FUZZFORMER_HAS_OPENGL
  if (opengl_initialized_) {
    return true;
  }

  glfwSetErrorCallback(glfw_error_callback);

  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

  GLFWwindow* window = glfwCreateWindow(width, height, "FuzzFormer Attention Visualization", nullptr, nullptr);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  OpenGLContext* ctx = new OpenGLContext();
  ctx->window = window;
  ctx->width = width;
  ctx->height = height;
  ctx->attention_width = 0;
  ctx->attention_height = 0;

  glfwSetWindowUserPointer(window, ctx);

  glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  opengl_context_ = ctx;
  opengl_initialized_ = true;
  return true;
#else
  (void)width;
  (void)height;
  return false;
#endif
}

void VisualRenderer::render_opengl_frame() {
#ifdef FUZZFORMER_HAS_OPENGL
  if (!opengl_initialized_ || !opengl_context_) {
    return;
  }

  OpenGLContext* ctx = static_cast<OpenGLContext*>(opengl_context_);
  GLFWwindow* window = ctx->window;

  glfwPollEvents();

  if (glfwWindowShouldClose(window)) {
    shutdown_opengl();
    return;
  }

  glClear(GL_COLOR_BUFFER_BIT);

  // Render attention heatmap as a simple quad
  if (!ctx->attention_data.empty() && ctx->attention_width > 0 && ctx->attention_height > 0) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Draw attention matrix as colored quads
    float cell_width = 2.0f / ctx->attention_width;
    float cell_height = 2.0f / ctx->attention_height;

    for (int y = 0; y < ctx->attention_height; ++y) {
      for (int x = 0; x < ctx->attention_width; ++x) {
        float value = ctx->attention_data[y * ctx->attention_width + x];
        
        // Map value to color (blue -> green -> red)
        float r = std::max(0.0f, (value - 0.5f) * 2.0f);
        float g = value < 0.5f ? value * 2.0f : 2.0f - value * 2.0f;
        float b = std::max(0.0f, (0.5f - value) * 2.0f);

        glColor3f(r, g, b);

        float x1 = -1.0f + x * cell_width;
        float y1 = -1.0f + y * cell_height;
        float x2 = x1 + cell_width;
        float y2 = y1 + cell_height;

        glBegin(GL_QUADS);
        glVertex2f(x1, y1);
        glVertex2f(x2, y1);
        glVertex2f(x2, y2);
        glVertex2f(x1, y2);
        glEnd();
      }
    }
  }

  glfwSwapBuffers(window);
#else
  // OpenGL not available
#endif
}

void VisualRenderer::shutdown_opengl() {
#ifdef FUZZFORMER_HAS_OPENGL
  if (opengl_initialized_ && opengl_context_) {
    OpenGLContext* ctx = static_cast<OpenGLContext*>(opengl_context_);
    if (ctx->window) {
      glfwDestroyWindow(ctx->window);
    }
    delete ctx;
    opengl_context_ = nullptr;
    opengl_initialized_ = false;
    glfwTerminate();
  }
#else
  if (opengl_initialized_) {
    opengl_context_ = nullptr;
    opengl_initialized_ = false;
  }
#endif
}

void VisualRenderer::render_attention(const torch::Tensor& attention) {
#ifdef FUZZFORMER_HAS_TORCH
  if (!initialized_) {
    return;
  }

  if (attention.dim() != 2) {
    std::cerr << "Warning: Expected 2D attention matrix, got " << attention.dim() << "D\n";
    return;
  }

  const auto height = static_cast<int>(attention.size(0));
  const auto width = static_cast<int>(attention.size(1));

  std::vector<std::vector<float>> matrix(height);
  auto cpu_attention = attention.cpu().contiguous();
  auto accessor = cpu_attention.accessor<float, 2>();

  for (int i = 0; i < height; ++i) {
    matrix[i].resize(width);
    for (int j = 0; j < width; ++j) {
      matrix[i][j] = accessor[i][j];
    }
  }

  render_attention_heatmap(matrix);

  // Update OpenGL context if available
#ifdef FUZZFORMER_HAS_OPENGL
  if (opengl_initialized_ && opengl_context_) {
    OpenGLContext* ctx = static_cast<OpenGLContext*>(opengl_context_);
    ctx->attention_data.clear();
    ctx->attention_data.reserve(height * width);
    ctx->attention_width = width;
    ctx->attention_height = height;
    
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        ctx->attention_data.push_back(matrix[i][j]);
      }
    }
  }
#endif
#else
  std::cout << "VisualRenderer: libtorch not available, skipping render\n";
#endif
}

}  // namespace fuzzformer

