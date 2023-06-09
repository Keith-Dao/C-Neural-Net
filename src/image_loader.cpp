#include "image_loader.hpp"
#include "utils.hpp"

using namespace loader;
#pragma region Image loader
const preprocessingFunctions ImageLoader::standardPreprocessing{
    utils::normaliseImage, utils::flatten};

#pragma endregion Image loader