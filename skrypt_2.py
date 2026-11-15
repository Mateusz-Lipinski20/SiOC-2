import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Nowy, osobny katalog na plansze do Ćwiczenia nr 2 (Demozaikowanie)
OUTPUT_DIR = "sprawozdanie_2_zdjecia"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. DEFINICJE FUNKCJI INTERPOLUJĄCYCH 1D (Z ĆWICZENIA NR 1)
def kernel_pi(x):
    """Funkcja prostokątna Pi(x) - najbliższy sąsiad."""
    return np.where((x >= -0.5) & (x < 0.5), 1.0, 0.0)


def kernel_lambda(x):
    """Funkcja trójkątna Lambda(x) - interpolacja liniowa."""
    return np.where(np.abs(x) <= 1.0, 1.0 - np.abs(x), 0.0)


def kernel_keys(x, a=-0.5):
    """Funkcja sześcienna Keysa (a = -0.5)."""
    abs_x = np.abs(x)
    res = np.zeros_like(abs_x, dtype=np.float64)

    m1 = abs_x <= 1.0
    x1 = abs_x[m1]
    res[m1] = (a + 2.0) * (x1 ** 3) - (a + 3.0) * (x1 ** 2) + 1.0

    m2 = (abs_x > 1.0) & (abs_x < 2.0)
    x2 = abs_x[m2]
    res[m2] = a * (x2 ** 3) - 5.0 * a * (x2 ** 2) + 8.0 * a * x2 - 4.0 * a

    return res


KERNELS = {
    'pi': (kernel_pi, 1, 'Najbliższy sąsiad Π(x)'),
    'lambda': (kernel_lambda, 1, 'Liniowa Λ(x)'),
    'keys': (kernel_keys, 2, 'Sześcienna Keys')
}


# 2. DEFINICJE MASEK FILTRÓW CFA (BAYER 2x2 ORAZ FUJI X-TRANS 6x6)
BAYER_PATTERN = np.array([
    ['G', 'R'],
    ['B', 'G']
])

XTRANS_PATTERN = np.array([
    ['G', 'B', 'R', 'G', 'R', 'B'],
    ['R', 'G', 'G', 'B', 'G', 'G'],
    ['B', 'G', 'G', 'R', 'G', 'G'],
    ['G', 'R', 'B', 'G', 'B', 'R'],
    ['B', 'G', 'G', 'R', 'G', 'G'],
    ['R', 'G', 'G', 'B', 'G', 'G']
])


def create_cfa_masks(shape, pattern):
    """Tworzy trójwymiarową maskę binarną (H, W, 3) dla kanałów R, G, B."""
    H, W = shape[:2]
    pH, pW = pattern.shape
    # Powielenie wzorca (kafelkowanie) na pełny wymiar obrazu
    tiled = np.tile(pattern, (H // pH + 1, W // pW + 1))[:H, :W]

    mask_r = (tiled == 'R').astype(np.float64)
    mask_g = (tiled == 'G').astype(np.float64)
    mask_b = (tiled == 'B').astype(np.float64)

    return np.stack([mask_r, mask_g, mask_b], axis=-1)


def simulate_cfa(img, pattern):
    """Symuluje rejestrację obrazu przez matrycę CMOS z filtrem CFA."""
    masks = create_cfa_masks(img.shape, pattern)
    raw_rgb = img.astype(np.float64) * masks
    return raw_rgb.astype(np.uint8), masks


# 3. SILNIK INTERPOLACJI 1D DO DEMOZAIKOWANIA (SPLOT SEPAROWALNY)
def convolve_1d_axis0(signal_2d, weights_1d):
    """Wykonuje splot 1D wzdłuż osi 0 (wierszy/kolumn) z odbiciem brzegowym."""
    radius = len(weights_1d) // 2
    padded = np.pad(signal_2d, ((radius, radius), (0, 0)), mode='reflect')
    out = np.zeros_like(signal_2d, dtype=np.float64)
    H = signal_2d.shape[0]

    for k, w in enumerate(weights_1d):
        if w != 0.0:
            out += padded[k: k + H, :] * w
    return out


def interpolate_channel_separable(raw_channel, mask_channel, method='keys', step=2.0):
    """
    Rekonstruuje pojedynczy kanał koloru metodą separowalnej interpolacji 1D
    (najpierw wzdłuż wierszy, potem wzdłuż kolumn).
    """
    kernel_func, R_base, _ = KERNELS[method]

    # Dla metody najbliższego sąsiada używamy minimalnego promienia pokrycia dziur
    if method == 'pi':
        max_dist = int(np.ceil(step))
        offsets = np.arange(-max_dist, max_dist + 1, dtype=np.float64)
        # Lekka asymetria zapobiegająca remisom odległości
        weights = kernel_pi(offsets / (max(step, 2.0) * 1.05) - 1e-6)
    else:
        max_dist = int(np.ceil(R_base * step))
        offsets = np.arange(-max_dist, max_dist + 1, dtype=np.float64)
        weights = kernel_func(offsets / step)

    # Krok 1: Interpolacja 1D wzdłuż osi pionowej (oś 0)
    num_y = convolve_1d_axis0(raw_channel, weights)
    den_y = convolve_1d_axis0(mask_channel, weights)

    # Krok 2: Transpozycja i interpolacja 1D wzdłuż osi poziomej (oś 1)
    num_xy = convolve_1d_axis0(num_y.T, weights).T
    den_xy = convolve_1d_axis0(den_y.T, weights).T

    reconstructed = num_xy / (den_xy + 1e-8)

    # W węzłach, gdzie piksel był oryginalnie zmierzony przez matrycę, zachowujemy oryginał
    reconstructed = np.where(mask_channel > 0.5, raw_channel, reconstructed)
    return reconstructed


def demosaic_image(raw_rgb, masks, cfa_type='bayer', method='keys'):
    """Główna procedura demozaikowania obrazu RGB dla wybranego filtru CFA."""
    out = np.zeros_like(raw_rgb, dtype=np.float64)

    # Dobór kroku siatki (odległości między znanymi próbkami w poszczególnych kanałach)
    if cfa_type == 'bayer':
        # W Bayerze R i B występują co 2 piksele, a G jest gęstsze (szachownica co sqrt(2))
        steps = [2.0, 1.45, 2.0]
    else:
        # W X-Trans R i B mają odstępy do 3 pikseli, a G tworzy bloki 2x2 i pojedyncze punkty
        steps = [2.6, 1.6, 2.6]

    for ch in range(3):
        out[:, :, ch] = interpolate_channel_separable(
            raw_rgb[:, :, ch].astype(np.float64),
            masks[:, :, ch],
            method=method,
            step=steps[ch]
        )

    return np.clip(out, 0, 255).astype(np.uint8)


# 4. FUNKCJE POMOCNICZE (WCZYTYWANIE KADRU I METRYKI MSE/PSNR)
def load_cropped_image(filename="Pasikonik.jpg", crop_size=200):
    """Wczytuje obraz i wycina kadr 200x200 px wycentrowany na głowie pasikonika."""
    paths_to_check = [filename, os.path.join("images", filename)]
    for path in paths_to_check:
        if os.path.exists(path):
            full_img = np.array(Image.open(path))
            H, W = full_img.shape[:2]
            half = crop_size // 2
            cy, cx = (int(H * 0.52), int(W * 0.30)) if "Pasikonik" in filename else (int(H * 0.34), int(W * 0.58))
            y1 = max(0, min(cy - half, H - crop_size))
            x1 = max(0, min(cx - half, W - crop_size))
            return full_img[y1: y1 + crop_size, x1: x1 + crop_size]

    # Obraz zastępczy w razie braku pliku
    y, x = np.meshgrid(np.linspace(-3, 3, crop_size), np.linspace(-3, 3, crop_size))
    ch1 = (np.sin(7 * (x ** 2 + y ** 2)) + 1.0) * 127.5
    ch2 = (np.cos(5 * x) * np.sin(5 * y) + 1.0) * 127.5
    ch3 = np.where((x ** 2 + y ** 2) < 4.0, 220.0, 40.0)
    return np.stack([ch1, ch2, ch3], axis=-1).astype(np.uint8)


def compute_metrics(orig, proc, margin=4):
    """Oblicza MSE i PSNR (z pominięciem wąskiego marginesu brzegowego)."""
    if margin > 0:
        orig = orig[margin:-margin, margin:-margin]
        proc = proc[margin:-margin, margin:-margin]
    mse = np.mean((orig.astype(np.float64) - proc.astype(np.float64)) ** 2)
    psnr = float('inf') if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)
    return mse, psnr


def plot_cfa_channels(raw_rgb, title_prefix, filename, zoom_slice=slice(60, 110)):
    """Generuje planszę z kanałami R, G, B oraz złożeniem RGB (jak Rys. 1 i 2 w instrukcji)."""
    crop_raw = raw_rgb[zoom_slice, zoom_slice]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    titles = ['red', 'green', 'blue', 'RGB']

    for ch in range(3):
        single_ch = np.zeros_like(crop_raw)
        single_ch[:, :, ch] = crop_raw[:, :, ch]
        axes[ch].imshow(single_ch)
        axes[ch].set_title(titles[ch], fontsize=11)
        axes[ch].axis('off')

    axes[3].imshow(crop_raw)
    axes[3].set_title('RGB', fontsize=11)
    axes[3].axis('off')

    plt.suptitle(title_prefix, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    img_orig = load_cropped_image("Pasikonik.jpg", crop_size=200)

    # 1. Symulacja matrycy Bayera i X-Trans
    raw_bayer, masks_bayer = simulate_cfa(img_orig, BAYER_PATTERN)
    raw_xtrans, masks_xtrans = simulate_cfa(img_orig, XTRANS_PATTERN)

    # Wygenerowanie Rysunku 1 i Rysunku 2 z instrukcji
    plot_cfa_channels(raw_bayer, "Obrazy RAW kanałów R, G, B oraz mozaika Bayer CFA (2x2)", "01_mozaika_bayer.png")
    plot_cfa_channels(raw_xtrans, "Obrazy RAW kanałów R, G, B oraz mozaika Fuji X-Trans (6x6)", "02_mozaika_xtrans.png")

    # 2. Demozaikowanie Bayer CFA dla trzech funkcji interpolacji
    res_bayer = {}
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    axes[0, 0].imshow(img_orig)
    axes[0, 0].set_title("Oryginał (200x200 px)", fontsize=10)
    axes[0, 0].axis('off')
    axes[1, 0].imshow(raw_bayer)
    axes[1, 0].set_title("Mozaika Bayer CFA", fontsize=10)
    axes[1, 0].axis('off')

    for idx, m in enumerate(['pi', 'lambda', 'keys']):
        _, _, label = KERNELS[m]
        rec = demosaic_image(raw_bayer, masks_bayer, cfa_type='bayer', method=m)
        res_bayer[m] = rec
        mse, psnr = compute_metrics(img_orig, rec)

        diff_map = np.clip(np.abs(img_orig.astype(float) - rec.astype(float)) * 5.0, 0, 255).astype(np.uint8)
        axes[0, idx + 1].imshow(rec)
        axes[0, idx + 1].set_title(f"{label}\nMSE: {mse:.2f} | PSNR: {psnr:.2f} dB", fontsize=9)
        axes[0, idx + 1].axis('off')

        axes[1, idx + 1].imshow(diff_map)
        axes[1, idx + 1].set_title("Błąd (|Δ| x5)", fontsize=9)
        axes[1, idx + 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_demozaikowanie_bayer.png"), dpi=150)
    plt.close()

    # 3. Demozaikowanie Fuji X-Trans dla trzech funkcji interpolacji
    res_xtrans = {}
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    axes[0, 0].imshow(img_orig)
    axes[0, 0].set_title("Oryginał (200x200 px)", fontsize=10)
    axes[0, 0].axis('off')
    axes[1, 0].imshow(raw_xtrans)
    axes[1, 0].set_title("Mozaika Fuji X-Trans", fontsize=10)
    axes[1, 0].axis('off')

    for idx, m in enumerate(['pi', 'lambda', 'keys']):
        _, _, label = KERNELS[m]
        rec = demosaic_image(raw_xtrans, masks_xtrans, cfa_type='xtrans', method=m)
        res_xtrans[m] = rec
        mse, psnr = compute_metrics(img_orig, rec)

        diff_map = np.clip(np.abs(img_orig.astype(float) - rec.astype(float)) * 5.0, 0, 255).astype(np.uint8)
        axes[0, idx + 1].imshow(rec)
        axes[0, idx + 1].set_title(f"{label}\nMSE: {mse:.2f} | PSNR: {psnr:.2f} dB", fontsize=9)
        axes[0, idx + 1].axis('off')

        axes[1, idx + 1].imshow(diff_map)
        axes[1, idx + 1].set_title("Błąd (|Δ| x5)", fontsize=9)
        axes[1, idx + 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_demozaikowanie_xtrans.png"), dpi=150)
    plt.close()