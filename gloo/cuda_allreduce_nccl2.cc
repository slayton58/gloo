/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/cuda_allreduce_nccl2.h"

#include "gloo/broadcast_one_to_all.h"
#include "gloo/cuda_private.h"

#include <unordered_map>

namespace gloo {

namespace {

// Creating NCCL communicators is expensive. So we cache and reuse them.
static std::shared_ptr<NCCLCommList> getCachedCommList(
    const std::shared_ptr<Context>& context,
    const std::vector<int> localDevices)
{
  static thread_local std::unordered_map<std::string, std::shared_ptr<NCCLCommList> >
    commLists;

  // generate key
  const int numDevices = localDevices.size();
  std::string key = std::to_string(context->size) + ' ' +
    std::to_string(context->rank);
  for (auto i = 0; i < numDevices; ++i) {
    key += ' ' + std::to_string(localDevices[i]);
  }

  // get or create CommList
  {
    static std::mutex m;
    //printf("looking for commlist %s from %d\n", key.c_str(), context->rank);
    // std::lock_guard<std::mutex> lock(m);
    if (!commLists[key]) {
      commLists[key] = std::make_shared<NCCLCommList>(context, localDevices);
    }
  }

  const auto commList = commLists[key];
  GLOO_ENFORCE_NE(commList.get(), (void*)nullptr);
  return commList;
}

} // namespace

NCCLCommList::NCCLCommList(const std::shared_ptr<Context>& context,
    const std::vector<int> localDevices) {
  // generate unique ID on root node
  ncclUniqueId *id = new ncclUniqueId;
  std::vector<char*> ids;
  ids.push_back(id->internal);
  if (context->rank == 0) {
    // std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    //printf("getting NCCL unique ID from rank %d\n", context->rank);
    NCCL_CHECK(ncclGetUniqueId(id));
    //printf("generated id: %s\n", id->internal);
  }

  //printf("(%d) current id is %s\n", context->rank, (char *)id->internal);

  // broadcast ID to other nodes
  //printf("bcasting id to all other ranks (%d)\n", context->rank);
  BroadcastOneToAll<char>(context, ids, NCCL_UNIQUE_ID_BYTES).run();

  //printf("(%d) current id is now %s\n", context->rank, id->internal);
  // create comms
  // FIXME currently, we assume all ranks use the same number of devices
  const int numDevices = localDevices.size();
	// num_ranks * num_devices_per_rank
  const int ncclSize = context->size * numDevices;
  // rank_id * num_devices_per_rank
  const int ncclRankStart = context->rank * numDevices;

  //printf("rank %d initializing %d ranks of %d total from %d with id: %p\n", context->rank, numDevices, ncclSize, ncclRankStart, id->internal);
  comms.reserve(numDevices);
  {
    // std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    NCCL_CHECK(ncclGroupStart());
    for (auto i = 0; i < numDevices; ++i) {
      CudaDeviceScope scope(localDevices[i]);
      NCCL_CHECK(ncclCommInitRank(&comms[i], ncclSize, *id,
                 ncclRankStart + i));
    }
    NCCL_CHECK(ncclGroupEnd());
  }
  //printf("rank %d all done\n", context->rank);
}

NCCLCommList::~NCCLCommList() {
  for (auto i = 0; i < comms.size(); ++i) {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    ncclCommDestroy(comms[i]);
  }
}

template <typename T>
CudaAllreduceNccl2<T>::CudaAllreduceNccl2(
  const std::shared_ptr<Context>& context,
  const std::vector<T*>& ptrs,
  const int count,
  const std::vector<cudaStream_t>& streams)
    : Algorithm(context),
      count_(count),
      bytes_(count_ * sizeof(T)),
      synchronizeDeviceOutputs_(streams.size() == 0),
      fn_(CudaReductionFunction<T>::sum) {
  auto newStream = true;
  if (streams.size() > 0) {
    GLOO_ENFORCE_EQ(streams.size(), ptrs.size());
    newStream = false;
  }

  for (auto i = 0; i < ptrs.size(); i++) {
    auto ptr = CudaDevicePointer<T>::create(ptrs[i], count_);
    if (newStream) {
      streams_.push_back(CudaStream(ptr.getDeviceID()));
    } else {
      streams_.push_back(CudaStream(ptr.getDeviceID(), streams[i]));
    }
    devicePtrs_.push_back(std::move(ptr));
  }

#if 0
  // Generate unique ID on root node
  ncclUniqueId id;
  std::vector<int8_t*> ids;
  ids.push_back((int8_t*)id.internal);
  if (context->rank == 0) {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    ncclGetUniqueId(&id);
  }

  // Broadcast ID to other nodes
  BroadcastOneToAll<int8_t>(context, ids, NCCL_UNIQUE_ID_BYTES).run();

  // send localDevices to all nodes
  const int localDevices = ptrs.size();
  std::vector<int> numDevices(context->size);
  std::vector<std::vector<int*>> numDevicesRefs(context->size);
  for (int i=0; i<context->size; i++) {
    numDevicesRefs[i].push_back(&numDevices[i]);
    numDevices[i] = (i == context->rank) ? localDevices : -1;
    BroadcastOneToAll<int>(context, numDevicesRefs[i], 1, i).run();
  }

  // Initialize nccl comms
  int ncclSize = 0;
  int ncclRank = 0;
  for (int i=0; i<context->size; i++) {
    ncclSize += numDevices[i];
    if (i < context->rank)
      ncclRank += numDevices[i];
  }
  comms_.resize(localDevices);
  {
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    NCCL_CHECK(ncclGroupStart());
    for (int i=0; i<localDevices; i++) {
      CUDA_CHECK(cudaSetDevice(devicePtrs_[i].getDeviceID()));
      NCCL_CHECK(ncclCommInitRank(&comms_[i], ncclSize, id, ncclRank + i));
    }
    NCCL_CHECK(ncclGroupEnd());
  }
#else
	// assemble list of local devices
  std::vector<int> localDevices(ptrs.size());
  for (auto i=0; i < devicePtrs_.size(); ++i) {
		localDevices[i] = devicePtrs_[i].getDeviceID();
	}
	commList_ = getCachedCommList(context, localDevices);
#endif
}

template <typename T>
void CudaAllreduceNccl2<T>::run() {
  {
    //printf("running allreduce on rank\n");
    std::lock_guard<std::mutex> lock(CudaShared::getMutex());
    NCCL_CHECK(ncclGroupStart());
    for (int i=0; i<devicePtrs_.size(); i++) {
      NCCL_CHECK(ncclAllReduce(
            (const void*)(*devicePtrs_[i]), (void*)(*devicePtrs_[i]),
            count_, nccl::ncclTypeWrapper<T>::type, ncclSum, commList_->comms[i], *streams_[i]));
    }
    NCCL_CHECK(ncclGroupEnd());
    //printf("rankall done\n");
  }

  for (int i=0; i<devicePtrs_.size(); i++)
    CUDA_CHECK(cudaStreamSynchronize(*streams_[i]));
}

#if 0
template <typename T>
CudaAllreduceNccl2<T>::~CudaAllreduceNccl2() {
  std::lock_guard<std::mutex> lock(CudaShared::getMutex());
  for (auto& comm : comms_)
    ncclCommDestroy(comm);
}
#endif

// Instantiate templates
#define INSTANTIATE_TEMPLATE(T)                                         \
template class CudaAllreduceNccl2<T>;

INSTANTIATE_TEMPLATE(int8_t);
INSTANTIATE_TEMPLATE(int32_t);
INSTANTIATE_TEMPLATE(int64_t);
INSTANTIATE_TEMPLATE(uint64_t);
INSTANTIATE_TEMPLATE(float);
INSTANTIATE_TEMPLATE(double);
INSTANTIATE_TEMPLATE(float16);

} // namespace gloo
