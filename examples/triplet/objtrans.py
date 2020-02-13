import bpy
for i in range(1, 51):
	override = {'selected_bases': list(bpy.context.scene.object_bases)}
	bpy.ops.object.delete(override)
	dir_from='/Users/yidawang/Downloads/temp/motorbike/'+str(i)+'/'+str(i)+'.obj'
	dir_to='/Users/yidawang/Downloads/collection/motorbike/'+str(i)+'/'+str(i)+'.obj'
	glob_type=str(i)+'.obj;'+str(i)+'.mtl'
	scene = bpy.context.scene
	lamp_data = bpy.data.lamps.new(name="New Lamp", type='HEMI')
	lamp_object = bpy.data.objects.new(name="New Lamp", object_data=lamp_data)
	scene.objects.link(lamp_object)
	lamp_object.location = (0, 0, 1)
	lamp_object.select = True
	scene.objects.active = lamp_object
	bpy.ops.import_scene.obj(filepath=dir_from,filter_glob=glob_type)
	bpy.ops.object.shade_flat()
	bpy.ops.export_scene.obj(filepath=dir_to,filter_glob=glob_type)

