from functools import wraps

def TFpostprocessing(func):
	@wraps(func)
	def wrapper(net_in,net_out,meta):
		print('calling TFpostprocessing decorator')
		A = func(net_in,net_out,meta)
		return func(net_in,net_out,meta)
	return wrapper


def TFpreprocessing(func):
	@wraps(func)
	def wrapper(net_in,meta):
		print('calling TFpreprocessing decorator')
		return func(net_in,net_out,meta)
	return wrapper


def postprocessing(func):
	@wraps(func)
	def wrapper(net_in,net_out,meta):
		print('calling postprocessing decorator')
		return func(net_in,net_out,meta)		
	return wrapper


def preprocessing(func):
	@wraps(func)
	def wrapper(net_in,meta):
		print('calling preprocessing decorator')
		return func(net_in,meta)
	return wrapper