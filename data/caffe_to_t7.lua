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

local base = '~/models/VGGNet/VOC0712/SSD_300x300/'
local model = loadcaffe.load(base..'deploy.prototxt', base..'VGG_VOC0712_SSD_300x300_iter_120000.caffemodel', 'nn')
-- model:remove() -- Remove the softmax at the end
-- assert(torch.isTypeOf(model:get(#model), nn.Linear))
-- model:evaluate()
-- model:float()

torch.save('weights/VOC0712.t7',model)
