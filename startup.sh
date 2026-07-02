#!/bin/bash
# Launch iris-main in background, wait for IRIS to be ready, then compile RunScript.mac
/iris-main &
IRIS_PID=$!

echo "Waiting for IRIS to be ready..."
for i in $(seq 1 30); do
    if echo "halt" | iris session IRIS -U USER > /dev/null 2>&1; then
        break
    fi
    sleep 2
done

echo "Compiling RunScript.mac..."
echo 'do $System.OBJ.Load("/home/irisowner/dev/src/RunScript.mac","ck") halt' \
    | iris session IRIS -U USER

echo "Ready."
wait $IRIS_PID
