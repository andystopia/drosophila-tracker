filename = "2022-06-10_1_UL.water-vapor_wild-type-vF_LL.water-vapor_wild-type_vM_UR.no-input_wild-type-vF_LR.no-input_wild-type-vM.mp4"

split = filename.split(".")

date, ul, ll, ur, lr, ext = split

quads = [ul, ll, ur, lr]

print(date)
for quad in quads:
    print(f"{date}.{quad}")