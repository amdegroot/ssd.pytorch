require 'torch'
require 'nn'
require 'loadcaffe'

--[[MIT License

Copyright (c) 2017 Justin Johnson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
]]--

-- local base = '~/models/VGGNet/VOC0712/SSD_300x300/'
-- base..
 -- base..
model = loadcaffe.load('../../weights/VGG_ILSVRC_16_layers_fc_reduced_deploy.prototxt','../weights/VGG_ILSVRC_16_layers_fc_reduced.caffemodel', 'nn')
-- model:remove() -- Remove the softmax at the end
-- assert(torch.isTypeOf(model:get(#model), nn.SpatialConvolution())) -- reduced version w/o linear layers
-- model:evaluate()
-- model:float()

torch.save('../../weights/vgg16_fc_reduced.t7',model)
