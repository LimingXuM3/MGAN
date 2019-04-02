-- requirement
require 'nn'
require 'nngraph'
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

-- options
cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:option( '--input', 'none', 'Input image' )
cmd:option( '--mask', 'none', 'Mask image')
cmd:option( '--maxdim', 512, 'Max size of input image')
cmd:option( '--gpu', true, 'Use GPU' )

local opt = cmd:parse(arg or {})
assert(opt.input ~= 'none')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Wraning: The input image and mask size must be same!')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++')

-- pre-process the image
function load_image(path)
   local img = image.load(path)
	
   if img:size(1) > 3 then
      img = img[{{1,3},{},{}}]
   end
	
   if img:size(1) == 2 then
      img = img[{{1},{},{}}]
   end
	
   if img:size(1) > 1 then
      img = image.rgb2y(img)
   end
	
   return img
end

-- load pre-trained MGAN
print( 'Loding model...' )
local data = torch.load( 'MGAN.t7' )
local inpaint_model = data.model
local datamean  = data.mean
inpaint_model:evaluate()

-- use GPU
if opt.gpu then
   require 'cunn'
   inpaint_model:cuda()
end

-- load data
local Input = image.load(opt.input)
local Mask = torch.Tensor()

-- inpaint the image with designated mask
if opt.mask ~='none' then
	print( 'Loding input image and corresponding mask...' )
	Mask = load_image(opt.mask)
	assert(Input:size(2) == Mask:size(2) and Input:size(3) == Mask:size(3))
else
-- inpaint the image with random mask
	print( 'Using random mask...' )
	Mask = torch.Tensor(1, Input:size(2), Input:size(3) ):fill(0)
	local nMasks = torch.random(2, 4)
	
	for i=1, nMasks do
		local mask_w = torch.random(32, 96)
		local mask_h = torch.random(32, 96)
		local x = torch.random(1, Input:size(3) - mask_w-1)
		local y = torch.random(1, Input:size(2) - mask_h-1)
		local Random = {{}, {y, y + mask_h}, {x, x + mask_w}}
		Mask[Random]:fill(1)
	end 
end

-- scale the image with an out of bound
local max_size = math.max(Input:size(2), Input:size(3))
if max_size > opt.maxdim then
	print('the size of input image is too large. Scaling...')
	Input = image.scale(Input, string.format('*%d/%d', opt.maxdim, max_size) )
	Mask = image.scale(Mask, string.format('*%d/%d', opt.maxdim, max_size) )
end

-- mask the input image
Input = image.scale(Input, torch.round(Input:size(3)/4)*4, torch.round(Input:size(2)/4)*4 )
Mask = image.scale(Mask, torch.round(Mask:size(3)/4)*4, torch.round(Mask:size(2)/4)*4 ):ge(0.2):float()
local Input_clone = Input:clone()
for j = 1,3 do 
	Input[j]:add(-datamean[j] ) 
end
Input:maskedFill(torch.repeatTensor(Mask:byte(),3,1,1), 0)

-- inpainting masked image
print('Start inpainting...')
local input = torch.cat(Input, Mask, 1)
input = input:reshape(1, input:size(1), input:size(2), input:size(3))
if opt.gpu then
	input = input:cuda()
end
local inpainted_image = inpaint_model:forward(input):float()[1]
local inpainted_result = Input_clone:cmul(torch.repeatTensor((1-Mask),3,1,1)) + inpainted_image:cmul(torch.repeatTensor(Mask,3,1,1))

-- save inpainted result
for j = 1,3 do 
	Input[j]:add(datamean[j]) 
end
image.save('damaged_image.png', Input)
image.save('inpainted_result.png', inpainted_result)

print('Testing Done!')
