"""Download MES historical data from IBKR — 60 days."""

import asyncio
import os
import pandas as pd


async def main():
    from ib_insync import IB, Future

    ib = IB()
    await ib.connectAsync("127.0.0.1", 7497, clientId=1)
    print("✅ Conectado a IBKR")

    contract = Future("MES", "202603", "CME")
    await ib.qualifyContractsAsync(contract)
    print(f"✅ Contrato: {contract.localSymbol}")

    all_bars = []

    # 12 bloques de 5 días = 60 días
    blocks = [
        ("5 D", ""),
        ("5 D", "20260301 23:59:59 US/Central"),
        ("5 D", "20260222 23:59:59 US/Central"),
        ("5 D", "20260215 23:59:59 US/Central"),
        ("5 D", "20260208 23:59:59 US/Central"),
        ("5 D", "20260201 23:59:59 US/Central"),
        ("5 D", "20260125 23:59:59 US/Central"),
        ("5 D", "20260118 23:59:59 US/Central"),
        ("5 D", "20260111 23:59:59 US/Central"),
        ("5 D", "20260104 23:59:59 US/Central"),
        ("5 D", "20251228 23:59:59 US/Central"),
        ("5 D", "20251221 23:59:59 US/Central"),
    ]

    for i, (duration, end_dt) in enumerate(blocks):
        label = end_dt[:8] if end_dt else "ahora"
        print(f"⏳ Bloque {i+1}/{len(blocks)}: hasta {label}...")
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_dt,
                durationStr=duration,
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=False,
                timeout=120,
            )
            if bars:
                all_bars.extend(bars)
                print(f"   ✅ {len(bars)} barras")
            else:
                print(f"   ⚠️ Sin datos")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        if i < len(blocks) - 1:
            print("   ⏸️ Pausa 11s...")
            await asyncio.sleep(11)

    if not all_bars:
        print("❌ No se descargaron datos.")
        ib.disconnect()
        return

    df = pd.DataFrame([{
        "time": b.date,
        "open": b.open,
        "high": b.high,
        "low": b.low,
        "close": b.close,
        "volume": b.volume,
    } for b in all_bars])

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)

    os.makedirs("data/historical", exist_ok=True)
    df.to_csv("data/historical/MES.csv", index=False)
    print(f"\n🎯 Guardadas {len(df)} barras en data/historical/MES.csv")
    print(f"   Desde: {df['time'].iloc[0]}")
    print(f"   Hasta: {df['time'].iloc[-1]}")
    print(f"   Días: ~{(df['time'].iloc[-1] - df['time'].iloc[0]).days}")

    ib.disconnect()
    print("\n✅ Listo. Ahora corre: py backtest.py")


if __name__ == "__main__":
    asyncio.run(main())