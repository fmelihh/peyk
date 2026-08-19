# inference-profiler

Bilgisayarının donanımını inceleyip **o donanımda çalışabilecek en uygun, güncel
yerel LLM modellerini** öneren cross-platform bir Python CLI aracı. Sadece rapor
üretir — model indirmez, kurmaz, çalıştırmaz.

- **Donanım tespiti** (CPU / RAM / GPU / VRAM), Linux öncelikli, macOS & Windows'ta da çalışır.
- **Çok kriterli değerlendirme**: hız, kalite, dil desteği (ör. Türkçe), bağlam uzunluğu, lisans.
- **Uygunluk katmanları**: `RAHAT ÇALIŞIR` / `ZORLAR` / `SIĞMAZ`.
- **Güncel katalog** + isteğe bağlı canlı doğrulama (Ollama registry & Hugging Face).

## Kurulum

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # geliştirme
# veya: pip install -e .          # sadece çalıştırma
```

## Kullanım

```bash
inference-profiler                          # varsayılan rapor (çevrimdışı katalog)
inference-profiler --use-case coding        # amaca göre ağırlıklandırma
inference-profiler --languages tr,en        # Türkçe destekleyen modelleri öne çıkar
inference-profiler --context 32768          # hedef bağlam uzunluğu
inference-profiler --cross-check            # Ollama/HF'den canlı boyut doğrulaması
inference-profiler --discover               # HF'deki trending GGUF modellerini keşfet
inference-profiler --json                   # makine-okunur çıktı
inference-profiler --markdown rapor.md      # Markdown rapor
```

### HuggingFace keşfi (`--discover`)

Curated katalogda **olmayan** güncel/trending GGUF modellerini HuggingFace'den
otomatik keşfeder: popüler GGUF repo'larını listeler, her quantization için
varyant (bölünmüş dosyaları toplayarak) çıkarır, repo adı ve etiketlerden
parametre/lisans/dil bilgisini türetir. Konuşma tanıma (ASR), embedding, görsel
gibi LLM olmayan GGUF'lar elenir. Bu modeller raporda **`HF keşif`** kaynağıyla
işaretlenir ve kalite puanları yalnızca parametre sayısından **kestirilmiştir**
(benchmark yok) — bilinçli kullanın.

Kısayol: `llm-fit` aynı komuttur.

## Nasıl çalışıyor?

1. `profiler/` donanımı normalize `HardwareProfile`e çevirir.
2. `sources/` model kataloğunu birleştirir (yerleşik JSON + isteğe bağlı Ollama/HF
   cross-check + isteğe bağlı HF keşfi).
3. `estimator.py` her varyant için bellek ihtiyacını (ağırlık + KV cache + overhead)
   ve kaba token/sn tahminini hesaplar.
4. `scoring.py` her modeli 5 kritere göre 0–100 puanlar.
5. `report.py` terminal / JSON / Markdown rapor üretir.

> **Not:** Bellek ve hız değerleri kaba tahmindir; gerçek sonuç backend,
> quantization ve bağlam uzunluğuna göre değişir. Ayrıntı: [`docs/design.md`](docs/design.md).

## Katalog güncelleme

Model listesi `src/inference_profiler/sources/data/catalog.json` içindedir.
Yeni model eklemek için bir varyant satırı ekleyin (params, quant, dosya boyutu,
bağlam, diller, lisans, kalite proxy). `--cross-check` boyutları canlı doğrular.

## Testler

```bash
pytest -q
```
