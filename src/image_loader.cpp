#include "image_loader.hpp"
#include "utils.hpp"

using namespace loader;

#pragma region Dataset batcher
int DatasetBatcher::size() const {
  return (this->data.size() + (this->dropLast ? 0 : this->batchSize - 1)) /
         this->batchSize;
}
#pragma endregion Dataset batcher

#pragma region Image loader
const preprocessingFunctions ImageLoader::standardPreprocessing{
    utils::normaliseImage, utils::flatten};
#pragma endregion Image loader