# Runbook: pods de AKS que fallan la readiness probe

> Documento sintético con fines educativos. Describe un procedimiento de ejemplo para un equipo de plataforma ficticio.

## Síntomas

- El Deployment muestra réplicas en estado `Running` pero `READY 0/1`.
- `kubectl describe pod` registra eventos `Readiness probe failed: HTTP probe failed with statuscode: 503`.
- El Service no tiene endpoints (`kubectl get endpoints <servicio>` devuelve `<none>`), por lo que el Ingress responde
  502 o 503 aunque los contenedores sigan vivos.
- Durante un despliegue, el rollout queda detenido con `Waiting for deployment rollout to finish: 0 of 3 updated
  replicas are available`.

## Diferencia entre liveness y readiness

La readiness probe decide si un pod recibe tráfico: cuando falla, Kubernetes lo quita de los endpoints del Service pero
no lo reinicia. La liveness probe decide si el contenedor debe reiniciarse. Un pod que falla solo la readiness queda
vivo y sin tráfico; por eso no aparece en `CrashLoopBackOff`.

## Diagnóstico paso a paso

1. Identificar los pods afectados: `kubectl get pods -n <namespace> -l app=<app> -o wide`.
2. Revisar los eventos del pod: `kubectl describe pod <pod> -n <namespace>` y buscar el código HTTP o el timeout de la
   probe.
3. Leer los logs de la aplicación: `kubectl logs <pod> -n <namespace> --since=15m`. Es frecuente que el endpoint
   `/health/ready` devuelva 503 porque una dependencia (base de datos, caché o un secreto de Key Vault) no está
   disponible.
4. Probar el endpoint desde dentro del pod: `kubectl exec -it <pod> -n <namespace> -- curl -sS -i
   localhost:8080/health/ready`.
5. Comparar la configuración de la probe con el tiempo real de arranque. Si la aplicación tarda 40 segundos en iniciar
   y `initialDelaySeconds` es 5 con `failureThreshold` 3 y `periodSeconds` 5, el pod nunca llega a estar listo a tiempo.
6. Verificar los recursos: un `CPU throttling` alto en el contenedor alarga el arranque y provoca timeouts de la probe.
   Revisar `kubectl top pod` y los límites de CPU del contenedor.

## Causas frecuentes y acciones

| Causa | Acción |
|---|---|
| Arranque lento de la aplicación | Agregar una `startupProbe` o subir `initialDelaySeconds`; no relajar la liveness probe para compensar |
| Dependencia caída (base de datos, caché) | Restaurar la dependencia; la readiness debe fallar mientras la dependencia no responda |
| Puerto o ruta de la probe incorrectos | Corregir `httpGet.port` y `httpGet.path` para que coincidan con el contenedor |
| Timeout de la probe demasiado bajo | Subir `timeoutSeconds` de 1 a 3 segundos si el endpoint hace verificaciones remotas |
| Límite de CPU insuficiente | Aumentar `resources.limits.cpu` o reducir el trabajo de arranque |

## Mitigación durante un incidente

- Si el problema empezó con un despliegue, hacer rollback: `kubectl rollout undo deployment/<app> -n <namespace>` y
  confirmar con `kubectl rollout status`.
- No eliminar la readiness probe para "forzar" tráfico: enviaría solicitudes a pods que no pueden atenderlas.
- Si hay réplicas sanas en otro node pool, escalar ese pool antes de reiniciar pods.

## Criterio de cierre

El incidente se considera resuelto cuando todas las réplicas muestran `READY 1/1` durante 15 minutos, el Service tiene
endpoints y la tasa de errores 5xx del Ingress vuelve a la línea base.
