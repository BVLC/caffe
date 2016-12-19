#ifdef CAFFE_MSL

#include "caffe/multinode/MslSync.hpp"

namespace caffe {

template<typename Dtype>
MslSync<Dtype>::MslSync(shared_ptr<Solver<Dtype> > root_solver)
        : solver(boost::make_shared<MslSolver<Dtype> >(root_solver))
        , initialized(false)
        , solver_thread_id(boost::this_thread::get_id())
        , snapshot_per_iters(root_solver->param().snapshot())
        , layers(root_solver->net()->layers())
        , net(root_solver->net())
        , net_params(root_solver->net()->learnable_params())
        , is_root(MSL::GetNodeId() == 0)
    {
        root_solver->param().set_disabled_update(true);
        if (!is_root) root_solver->param().clear_snapshot();
        if (!is_root) root_solver->param().set_snapshot_after_train(false);
        //if (!is_root) root_solver->param().clear_test_interval();

        if (root_solver->iter() == 0)
            root_solver->set_iter(1);

        for (int idx = 0; idx < layers.size(); idx++)
        {
            layers[idx]->bottom_vec = root_solver->net()->bottom_vecs()[idx];
            layers[idx]->top_vec = root_solver->net()->top_vecs()[idx];
        }

        layer_param_ids.resize(layers.size());

        bottom_pack_block_nums.resize(layers.size());
        bottom_unpack_block_nums.resize(layers.size());
        top_pack_block_nums.resize(layers.size());
        top_unpack_block_nums.resize(layers.size());

        for (int layer_id = 0; layer_id < layers.size(); layer_id++) {
            shared_ptr<Layer<Dtype> > layer = layers[layer_id];

            /* cache param ids */
            layer_param_ids[layer_id] = net->get_layer_learnable_param_ids(layer_id);

            /* cache bottom/top pack/unpack blocks nums */
            int bottom_size = layer->layer_param().bottom_size();
            int top_size = layer->layer_param().top_size();

            bottom_pack_block_nums[layer_id].resize(bottom_size, 0);
            bottom_unpack_block_nums[layer_id].resize(bottom_size, 0);

            top_pack_block_nums[layer_id].resize(top_size, 0);
            top_unpack_block_nums[layer_id].resize(top_size, 0);

            if (layer->layerOp->NumInputFeatureMaps())
            {
                for (int bottom_id = 0; bottom_id < bottom_size; ++bottom_id) {
                    FeatureMap *fm = layer->layerOp->InputFeatureMap(bottom_id);
                    bottom_pack_block_nums[layer_id][bottom_id] = fm->NumPackBlocks();
                    bottom_unpack_block_nums[layer_id][bottom_id] = fm->NumUnpackBlocks();
                }
            }

            if (layer->layerOp->NumOutputFeatureMaps())
            {
                for (int top_id = 0; top_id < top_size; ++top_id) {
                    FeatureMap *fm = layer->layerOp->OutputFeatureMap(top_id);
                    top_pack_block_nums[layer_id][top_id] = fm->NumPackBlocks();
                    top_unpack_block_nums[layer_id][top_id] = fm->NumUnpackBlocks();
                }
            }

            /* set owned_count and owned_offset for distributed weight update */
#ifdef DISTR_WEIGHT_UPDATE
            if (layer->layerOp->HasWeights()) {

                vector<int>& param_ids = layer_param_ids[layer_id];
                CHECK_NUM_WEIGHTS(layer, param_ids);

                for (int i = 0; i < param_ids.size(); ++i) {
                    int owned_count = layer->layerOp->Weights(i)->OwnedLen() * layer->layerOp->Weights(i)->WTSize();
                    int owned_offset = layer->layerOp->Weights(i)->OwnedStart() * layer->layerOp->Weights(i)->WTSize();

                    if (layer->layerOp->Weights(i)->LocalLen() * layer->layerOp->Weights(i)->WTSize() != net_params[param_ids[i]]->count()) {
                        LOG(FATAL) << "different local weigth count in CAFFE (" << net_params[param_ids[i]]->count() << ") "
                                   << "and MSL (" << layer->layerOp->Weights(i)->LocalLen() * layer->layerOp->Weights(i)->WTSize() << ")";
                    }
                    /*if (layer->layerOp->Weights(i)->LocalLen() * layer->layerOp->Weights(i)->WTSize() > net_params[param_ids[i]]->count()) {
                        owned_count -= (layer->layerOp->Weights(i)->LocalLen() * layer->layerOp->Weights(i)->WTSize() - net_params[param_ids[i]]->count());
                    }*/

                    net_params[param_ids[i]]->set_owned_count(owned_count);
                    net_params[param_ids[i]]->set_owned_offset(owned_offset);

                    LOG(INFO) << "layer " << layer->type()
                                 << ", weigth_idx " << i
                                 << ", MSL owned_cout " << layer->layerOp->Weights(i)->OwnedLen() * layer->layerOp->Weights(i)->WTSize()
                                 << ", CAFFE owned_count " << owned_count
                                 << ", CAFFE owned_offset " << owned_offset
                                 << ", MSL local_count " << layer->layerOp->Weights(i)->LocalLen() * layer->layerOp->Weights(i)->WTSize()
                                 << ", CAFFE local_count " << net_params[param_ids[i]]->count();
                }
            }
#endif /* DISTR_WEIGHT_UPDATE */
        }
    }

template<typename Dtype>
MslSync<Dtype>::~MslSync()
{
}

  INSTANTIATE_CLASS(MslSync);
} // namespace caffe

#endif /* CAFFE_MSL */
