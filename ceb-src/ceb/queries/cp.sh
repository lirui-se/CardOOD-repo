dri=(tpl018  tpl019  tpl025  tpl027  tpl040  tpl050  tpl072  tpl084  tpl091  tpl099  tpl100  tpl101  tpl102)
for d in ${dri}; do
    mkdir dsb_1/${d}
    for ((i=1;i<=100;i++));
    do
        cp dsb/${d}/${d}-${i}.pkl dsb_1/${d}/
    done
done
