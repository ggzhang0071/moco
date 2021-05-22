#img="taikiinoue45/mvtec:mvtec"
img="pytorch/pytorch"

nvidia-docker run --privileged=true  --workdir /git --name "moco"  -e DISPLAY --ipc=host -d --rm  -p 5513:8889  \
-v /mnt/work/git/moco/:/git/moco \
$img sleep infinity