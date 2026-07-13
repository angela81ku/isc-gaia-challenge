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

# Register Gemini LLM config in AI Hub if GEMINI_API_KEY is set
if [ -n "$GEMINI_API_KEY" ]; then
    echo "Registering Gemini LLM config in AI Hub..."
    /usr/irissys/bin/irispython /home/irisowner/dev/src/register_llm.py || \
        echo "Warning: AI Hub LLM registration skipped"
fi

echo "Ready."
wait $IRIS_PID
