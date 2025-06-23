train_l=(tpl072 tpl040 tpl101 tpl025 tpl099 tpl102 tpl084 tpl100 tpl091 tpl019)
test_l=(tpl018 tpl027 tpl050)


for e in ${train_l[@]}; do
    if [ -f train/$e -o -d train/$e -o -L train/$e ]; then
        rm -rf train/$e
    fi
    ln -s ../../dsb/$e train/$e
done

for e in ${test_l[@]}; do
    if [ -f test/$e -o -d test/$e -o -L test/$e ]; then
        rm -rf test/$e
    fi
    ln -s ../../dsb/$e test/$e
done

