# Değerlendirme sonuçları

- Tarih: 2026-10-12 12:01
- Mod: arama + LLM (--with-llm)
- Sohbet modeli: `gemini-3.8-flash` (yedek: `gemini-3.5-flash-lite`, düşünme seviyesi: `low`)
- Embedding modeli: `gemini-embedding-2` (768 boyut)
- Arama eşiği (SEARCH_MIN_SCORE): 0.63, top-k: 5
- Soru sayısı: 18

## Özet

| Ölçüt | Sonuç |
| --- | --- |
| hit@1 (ilk kaynak doğru) | 14/14 (%100) |
| hit@4 (ilk 4 kaynakta doğru) | 14/14 (%100) |
| Alakalı soruda eşiği geçme | 14/14 (%100) |
| Katalog dışı soruda eşikte elenme | 4/4 (%100) |
| Doğru araç + doğru slug | 2/2 (%100), 1 soru ölçülemedi |
| Yanıtta atıf bulunma | 13/13 (%100), 1 soru ölçülemedi |

## Soru bazında

| id | tür | soru | en iyi skor / kaynak | hit@1 | hit@4 | eşik | araç | atıf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k1 | katalog | Tost için hangi kaşar peyniri uygun? | 0.79 product:kasar-eski | ✓ | ✓ | ✓ | — | ✓ |
| k2 | katalog | Deri tulum içinde olgunlaşan peynir hangisi? | 0.80 product:tulum-peyniri | ✓ | ✓ | ✓ | — | ✓ |
| k3 | katalog | Domuz eti içermeyen salam var mı? | 0.81 product:dana-salam | ✓ | ✓ | ✓ | — | ✓ |
| k4 | katalog | Süzme yoğurttan yapılan sürülebilir bir peynir var mı? | 0.77 product:labne-peyniri | ✓ | ✓ | ✓ | — | ✓ |
| k5 | katalog | Balmumu içinde, doğrudan petekten bal satıyor musunuz? | 0.75 product:petek-bal | ✓ | ✓ | ✓ | — | ✓ |
| k6 | katalog | Fındık ezmesinde şeker var mı? | 0.81 product:findik-ezmesi | ✓ | ✓ | ✓ | — | ✓ |
| k7 | katalog | Hafif tütsülenmiş bir peynir önerir misiniz? | 0.76 product:cerkez-peyniri | ✓ | ✓ | ✓ | — | ✓ |
| s1 | sss | Siparişim kaç günde teslim edilir? | 0.80 faq:siparisim-kac-gunde-teslim-edilir | ✓ | ✓ | ✓ | — | ✓ |
| s2 | sss | Açılmış peynirleri buzdolabında nasıl saklamalıyım? | 0.83 faq:peynirler-acldktan-sonra-nasl-saklanmal | ✓ | ✓ | ✓ | — | ✓ |
| s3 | sss | Ambalajı açılmış bir ürünü iade edebilir miyim? | 0.80 faq:urunleri-iade-edebilir-miyim | ✓ | ✓ | ✓ | — | ✓ |
| s4 | sss | Siparişe hediye notu ekleyebilir miyim? | 0.83 faq:hediye-paketi-ve-not-ekleyebilir-miyim | ✓ | ✓ | ✓ | — | ✓ |
| f1 | fiyat | Naturel sızma zeytinyağı kaç TL? | 0.78 product:naturel-zeytinyagi | ✓ | ✓ | ✓ | ✓ | ✓ |
| f2 | stok | Bafra Kaymağı stokta var mı? | 0.82 product:kaymak | ✓ | ✓ | ✓ | ✓ | ✓ |
| f3 | fiyat | Eski Kaşar'ın fiyatı ne kadar? | 0.79 product:kasar-eski | ✓ | ✓ | ✓ | ölçülemedi | ölçülemedi |
| d1 | katalog_disi | Yarın İstanbul'da hava nasıl olacak? | 0.57 faq:musteri-hizmetlerine-nasl-ulasrm | — | — | elendi |  |  |
| d2 | katalog_disi | Python'da bir liste nasıl sıralanır? | 0.57 faq:siparisimi-iptal-edebilir-miyim | — | — | elendi |  |  |
| d3 | katalog_disi | Türkiye'nin başkenti neresidir? | 0.59 faq:hangi-kargo-firmasyla-gonderim-yapyorsunuz | — | — | elendi |  |  |
| d4 | katalog_disi | Hangi telefon modelini almalıyım? | 0.61 faq:urun-bozuk-ya-da-hasarl-geldi-ne-yapmalym | — | — | elendi |  |  |

hit@k: beklenen kaynak, benzerliğe göre sıralı ilk k *farklı* kaynak içinde. eşik: en az bir parça eşiği geçti (katalog dışı sorularda 'elendi' beklenen sonuçtur). ölçülemedi: sağlayıcı hatası; ilgili oran hesabına katılmaz.
