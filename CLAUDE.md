# Ürün Asistanı (RAG) — CLAUDE.md

Bu dosya projenin tek karar kaynağıdır. Her turun başında oku. Burada yazmayan bir
karar gerekiyorsa uydurma; raporda soru olarak bildir.

## 1. Amaç

Bir şarküteri e-ticaret sitesinin ürün kataloğu üzerinde çalışan, müşteri sorularını
katalogdan bulduğu bilgiyle yanıtlayan sohbet servisi. Portföy projesidir: hedef,
bir LLM servisinin mevcut bir iş sistemine nasıl entegre edildiğini temiz ve
okunabilir biçimde göstermek.

Gösterilecek konular: RAG, vektör veritabanı, prompt tasarımı, araç çağırma
(tool calling), LLM API entegrasyonu, Docker, test ve ölçüm.

## 2. Kapsam

Var:
- Ürün verisini parçalayıp embedding'leriyle saklayan içe aktarma komutu
- Anlamsal arama katmanı
- Kaynak gösteren, katalogda olmayan konuda "bilmiyorum" diyen sohbet API'si
- Araç çağırma: stok ve fiyat sorgusu veritabanından
- Tek sayfalık React sohbet arayüzü
- 15-20 soruluk değerlendirme seti ve ölçüm komutu
- Docker Compose ile tek komutla ayağa kalkma

Yok (ekleme):
- Kullanıcı girişi, üyelik, sepet, ödeme
- Gerçek siteye bağlantı ya da gerçek müşteri/sipariş verisi
- Canlıya alma, CI/CD
- LangChain / LlamaIndex gibi çatılar (akış elle yazılacak, okunur kalsın)

## 3. Yığın

- Python 3.12, Django 5, Django REST Framework
- PostgreSQL 16 + pgvector (Docker imajı: pgvector/pgvector), Django tarafında `pgvector` paketi
- React + Vite (TypeScript), tek sayfa
- Test: pytest + pytest-django
- Docker Compose: `db`, `backend`, `frontend`

## 4. LLM sağlayıcısı

- Varsayılan: Google Gemini API, ücretsiz katman. İkinci seçenek: Ollama (yerel).
- Kod sağlayıcıdan bağımsız olacak: `llm/base.py` içinde bir arayüz
  (`embed(texts)`, `chat(messages, tools)`), `llm/gemini.py` ve `llm/ollama.py` uygulamaları.
  Hangisinin kullanılacağı `.env` içindeki `LLM_PROVIDER` ile seçilir.
- Model adları koda gömülmez; `.env` içinde `CHAT_MODEL` ve `EMBED_MODEL` olarak durur.
  Gemini model adları sık değişiyor: adı yazmadan önce resmî dokümandan güncel ve
  ücretsiz katmanda olan sohbet ve embedding modelini doğrula.
- Embedding boyutu `EMBED_DIM` ayarından gelir; vektör alanı bu boyutla oluşturulur.
  Model değişirse yeni migration ve yeniden içe aktarma gerekir.
- Ücretsiz katmanda hız sınırı var: içe aktarma toplu (batch) çalışır, 429 hatasında
  bekleyip yeniden dener.
- API anahtarı yalnızca `.env` içinde durur. `.env` git'e girmez, `.env.example` girer.

## 5. Veri

- Katalog uydurma örnek veridir: `data/products.json`, yaklaşık 30 ürün
  (peynirler, tereyağı, zeytin, sucuk, bal vb.).
- Ürün alanları: ad, kategori, açıklama, içindekiler, alerjenler, net ağırlık,
  saklama koşulu, menşe, fiyat, stok adedi.
- Ek olarak `data/faq.md`: kargo, iade, saklama gibi 10-15 genel soru-cevap.
- Gerçek işletme adı, gerçek fiyat ya da gerçek müşteri verisi kullanılmaz.

## 6. Mimari

```
soru -> embedding -> pgvector'da en yakın K parça -> prompt (sistem + parçalar + soru)
     -> model -> (gerekirse araç çağrısı -> veritabanı -> model) -> yanıt + kaynaklar
```

Django uygulamaları:
- `catalog`: Product modeli, örnek veriyi yükleme
- `rag`: Chunk modeli (metin, kaynak, embedding), içe aktarma, arama
- `llm`: sağlayıcı arayüzü ve uygulamaları
- `chat`: sohbet uç noktası, prompt, araç tanımları

Kurallar:
- Fiyat ve stok embedding'e girmez; her zaman araç çağrısıyla veritabanından okunur
  (değişen veri ile sabit bilgi ayrımı).
- Yanıt yalnızca getirilen parçalara dayanır. Parçalarda yoksa model bilmediğini söyler.
- Her yanıt kullandığı kaynakların listesini döndürür.
- Yanıt dili Türkçe.

API:
- `POST /api/chat/` — gövde: `{ "message": str, "history": [...] }`
  yanıt: `{ "answer": str, "sources": [...], "tool_calls": [...] }`
- `GET /api/health/`

## 7. Turlar

Her tur ayrı bir Claude Code oturumudur. Tur bitmeden sonrakine geçme.

- Tur 1 — İskelet: Django projesi, Docker Compose, pgvector, modeller, örnek veri,
  `llm` arayüzü ve Gemini uygulaması, `ingest` komutu.
- Tur 2 — Arama ve sohbet: arama katmanı, prompt, `/api/chat/`, kaynak gösterme,
  "bilmiyorum" davranışı.
- Tur 3 — Araç çağırma: `get_price`, `check_stock` araçları ve model-araç döngüsü.
- Tur 4 — Arayüz: React sohbet sayfası, kaynakların ve araç çağrılarının gösterimi.
- Tur 5 — Ölçüm ve belge: `evaluate` komutu, değerlendirme seti, README
  (mimari şema, kurulum, ekran görüntüsü yerleri, ölçüm sonuçları), Ollama uygulaması.

## 8. Çalışma kuralları

- Testler gerçek API'ye istek atmaz; LLM katmanı sahte (fake) sağlayıcıyla test edilir.
- Kod ve yorumlar İngilizce, README ve kullanıcıya dönük metinler Türkçe.
- Ağır işleri (docker build, test, npm install) aynı anda değil sırayla çalıştır.
- Commit etme; commit'i Mert yapar.
- Tur sonunda en fazla 25 satırlık rapor: ne yapıldı, nasıl denenir, açık sorular.
- Ortam: Windows, PowerShell, Docker Desktop.

## 9. Karar günlüğü

- K-1: Çatı kullanılmayacak, RAG akışı elle yazılacak.
- K-2: Vektör deposu pgvector (ayrı vektör veritabanı yok).
- K-3: Varsayılan sağlayıcı Gemini ücretsiz katman, yedek Ollama.
- K-4: Katalog uydurma örnek veri.
- K-5: Fiyat ve stok yalnızca araç çağrısıyla okunur.
- K-6: Embed arayüzü `task` parametresi alır (`"document"` | `"query"`). gemini-embedding-2 `task_type` desteklemez; görev metin önekiyle verilir (sorgu: `task: search result | query: ...`, doküman: `title: ... | text: ...`). Önek değişince doküman vektörleri değişir: `ingest --force` gerekir.
- K-7: Her metin ayrı `Content` olarak gönderilir (gemini-embedding-2 düz string listesini tek bir vektöre birleştirir).
