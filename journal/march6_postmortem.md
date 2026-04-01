# Postmortem operativo — 6 de marzo de 2026 (futuros)

## Hallazgos cuantitativos (dataset local MES)

Se analizó `data/historical/MES.csv` en horario regular de NY (09:30–16:00 ET) para la semana del 2 al 6 de marzo de 2026.

- El **6 de marzo** tuvo una apertura muy agresiva en los primeros 15 minutos (**-23.75 pts**) pero para la primera hora ya había revertido a **+9.50 pts** desde la apertura.
- Ese patrón es de **whipsaw temprano**: el mercado primero rompe en una dirección y luego revierte rápido.
- En la muestra de marzo disponible en el repo (5 días), el patrón whipsaw apareció en **3 de 5 días (60%)**.

Interpretación: si el bot entra exactamente a las 09:30 con lógica de continuación, es fácil que quede atrapado en el primer impulso falso.

## Por qué “la configuración inicial” pudo verse mejor

Con un mercado de whipsaw, una configuración inicial más simple (menos filtros y más frecuencia) puede parecer mejor por tres razones:

1. **Captura más rebotes**: en rangos/reversiones rápidas, muchos filtros de “calidad” eliminan trades que sí funcionaban por mean reversion.
2. **Menos latencia por validaciones**: agregar pre-scan/filtros puede retrasar la entrada y convertir un buen precio en entrada tardía.
3. **Sin bloqueo de utilidad**: al no asegurar ganancia diaria, una sesión verde puede terminar roja (que es justo lo que reportaste).

## Hipótesis principal para tus pérdidas actuales

No parece un solo problema; probablemente es una combinación:

- **Entrada demasiado temprano (09:30) en días de whipsaw**.
- **Stack de filtros que sobre-restringe** o entra tarde.
- **Gestión de ganancias diaria insuficiente/tardía** (aunque ya hay lock dinámico en el código, puede necesitar tuning).

## Sugerencias accionables (ordenadas por impacto)

1. **Mover inicio de ejecución de entradas a 09:45 ET** (manteniendo observación desde 09:30).
2. **Mantener pre-scan solo como ranking, no como bloqueo duro** (evitar quedarte sin señales útiles).
3. **Aplicar regla de régimen de apertura**:
   - Si la vela/rango de 09:30–09:45 supera un umbral alto, operar solo pullbacks/reversión hasta 10:15.
4. **Lock de utilidad más temprano**:
   - Activación por umbral absoluto diario bajo (ej. +$75/+100) y retención agresiva en apertura (70–80%).
5. **A/B test por 10–15 sesiones**:
   - A: inicio 09:30.
   - B: inicio 09:45.
   - Mismos stops/targets y mismo tamaño.
   - Medir: win rate, profit factor, MAE inicial, drawdown antes de 10:00.

## Conclusión

Tu intuición es correcta: **sí vale la pena analizar histórico intradía para replicar lo que funcionó**. En los datos locales, el 6 de marzo encaja en un régimen donde la apertura inmediata fue inestable; por eso retrasar entradas y simplificar filtros duros debería mejorar consistencia.

## Estado actual de la configuración (auditoría rápida)

- **Interés compuesto: ACTIVO.** El tracker actualiza capital con cada trade y el sizing se calcula desde ese capital actualizado.
- **Ganancia no topada: ACTIVA.** El lock dinámico protege retrocesos desde el pico, pero no limita cuánto puede subir el P&L diario.
- **Sizing por riesgo porcentual: ACTIVO.** Futuros usa % de capital asignado y lo traduce a contratos con tope máximo configurable.
- **Nuevo control opcional para replicar mejor días tipo 6 de marzo:** `FUTURES_NY_ENTRY_DELAY_MIN` (0 = desactivado, 15 = empezar a operar 09:45 ET).
