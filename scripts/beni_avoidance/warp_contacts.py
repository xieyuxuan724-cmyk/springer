import warp as wp

@wp.kernel
def detect_contacts(count:wp.array[int],geom:wp.array[wp.vec2i],dist:wp.array[float],
                    world:wp.array[int],kind:wp.array[int],hit:wp.array[int],ground:wp.array[int]):
    i=wp.tid()
    if i<count[0] and dist[i]<=0.0:
        a=kind[geom[i][0]];b=kind[geom[i][1]]
        if (a==2 and (b==1 or b==3)) or (b==2 and (a==1 or a==3)):
            wp.atomic_max(hit,world[i],1)
        if (a==4 and b==1) or (b==4 and a==1):
            wp.atomic_max(ground,world[i],1)
