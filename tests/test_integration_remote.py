"""Integration tests for remote inference server.

These tests run against a live inference server and validate that generation
completes successfully for challenging prompts that often cause issues with
body part generation (complex poses, twisted limbs, unusual angles).

Requires: Server running at INFERENCE_URL (default: http://localhost:8000).
Use `docker compose up` or k8s deployment before running.
Outputs are saved to output/test/ for manual review.
"""

import base64
import os
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image

INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:8000")
GENERATION_TIMEOUT = 120.0  # 120s timeout for generation with upsampling (~20s upsample + ~70s gen)
OUTPUT_DIR = Path("output/test")
TEST_WIDTH = 768
TEST_HEIGHT = 1024


def save_test_output(img_base64: str, name: str, seed: int) -> Path:
    """Save generated image for manual review."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img_bytes = base64.b64decode(img_base64)
    img = Image.open(BytesIO(img_bytes))
    output_path = OUTPUT_DIR / f"{name}_seed{seed}.png"
    img.save(output_path)
    return output_path


CHALLENGING_POSE_PROMPTS = [
    pytest.param(
        "Revolved Half Moon Balance",
        """A woman standing on one leg, full body visible, the supporting foot planted firmly on the ground while the opposite leg extends straight backward at hip height. Her torso twists sideways toward the camera, one hand reaching down to touch the floor while the other arm stretches vertically upward. Spine visibly rotated, shoulders stacked unevenly, hips misaligned by design. Tight athletic clothing clearly showing leg separation, knee alignment, ankle angle, and the twist of the waist. Camera at waist height, slight three-quarter angle, clean studio lighting revealing exact limb positioning.""",
        id="revolved_half_moon",
    ),
    pytest.param(
        "One-Legged Crow Transition",
        """A woman balanced low to the ground in a yoga arm balance. Both hands planted flat on the floor, elbows bent at sharp angles, shoulders leaning forward. One knee rests against the upper arm while the opposite leg extends backward fully off the ground. Head slightly lifted, neck extended forward. Weight distribution clearly visible through shoulder compression and wrist angle. Full body in frame from a low side angle, emphasizing arm strain, bent joints, and asymmetry between legs.""",
        id="one_legged_crow",
    ),
    pytest.param(
        "Deep Backbend Dropback",
        """A woman standing upright mid-transition into a deep backbend. Knees slightly bent, hips pushed forward, spine arched dramatically backward. Head tilted fully behind her with face upside down relative to torso. Arms reaching behind toward the floor but not yet touching. Rib cage lifted, abdomen stretched, pelvis visibly angled forward. Shot from the side at chest height, strong directional lighting highlighting spinal curvature and torso deformation under tension.""",
        id="deep_backbend",
    ),
    pytest.param(
        "Twisted Seated Bind",
        """A woman seated on the ground with one leg folded under her and the other bent across her body. Torso twisted sharply in the opposite direction of the legs. One arm wraps behind her back while the other reaches around the front to clasp the wrist, forming a closed bind behind her torso. Shoulders uneven, spine corkscrewed. Camera positioned slightly above, looking down to emphasize overlapping limbs and hidden joints. Clear visibility of hand placement, elbow direction, and torso rotation.""",
        id="twisted_seated_bind",
    ),
    pytest.param(
        "Standing Split With Forward Fold",
        """A woman folded forward at the hips with her torso fully inverted, head hanging downward. One leg remains grounded while the other leg lifts straight upward into a vertical split behind her. Hands gripping the standing ankle for balance. Hips uneven, pelvis tilted, legs forming a sharp asymmetrical line. Camera directly from the side to expose hip misalignment, leg separation, knee locking, and foot orientation. Neutral background, sharp lighting, no motion blur.""",
        id="standing_split",
    ),
]


def server_available() -> bool:
    """Check if the inference server is reachable."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{INFERENCE_URL}/ready")
            if resp.status_code == 200:
                data = resp.json()
                return data.get("ready", False)
            print(f"Server returned status {resp.status_code}")
    except httpx.ConnectError as e:
        print(f"Cannot connect to {INFERENCE_URL}: {e}")
    except httpx.TimeoutException as e:
        print(f"Timeout connecting to {INFERENCE_URL}: {e}")
    except Exception as e:
        print(f"Error checking server availability: {type(e).__name__}: {e}")
    return False


@pytest.fixture(scope="module")
def client():
    """HTTP client with extended timeout for generation.

    Fails if server is not available - integration tests assume infrastructure is running.
    """
    if not server_available():
        pytest.fail(f"Inference server not available at {INFERENCE_URL}. Start with docker compose or k8s.")
    return httpx.Client(base_url=INFERENCE_URL, timeout=GENERATION_TIMEOUT)


class TestServerHealth:
    """Basic server health checks."""

    def test_health_endpoint(self, client: httpx.Client):
        """Verify /health endpoint returns OK."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_name" in data

    def test_ready_endpoint(self, client: httpx.Client):
        """Verify /ready endpoint shows models loaded."""
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True
        assert data["model_loaded"] is True
        assert data["text_encoder_loaded"] is True
        assert data["ae_loaded"] is True


class TestBasicGeneration:
    """Basic generation functionality tests."""

    def test_simple_prompt(self, client: httpx.Client):
        """Generate with a simple prompt."""
        resp = client.post(
            "/generate",
            json={
                "prompt": "A red apple on a white table",
                "width": 512,
                "height": 512,
                "num_steps": 4,
                "guidance": 1.0,
                "seed": 42,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "image_base64" in data
        assert data["seed"] == 42
        assert data["width"] == 512
        assert data["height"] == 512

        img_bytes = base64.b64decode(data["image_base64"])
        img = Image.open(BytesIO(img_bytes))
        assert img.size == (512, 512)

    def test_generation_with_upsampling(self, client: httpx.Client):
        """Generate with prompt upsampling enabled."""
        resp = client.post(
            "/generate",
            json={
                "prompt": "A cat",
                "width": 512,
                "height": 512,
                "num_steps": 4,
                "guidance": 1.0,
                "seed": 123,
                "upsample": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["prompt"]) > len("A cat"), "Upsampled prompt should be longer"

    def test_different_aspect_ratios(self, client: httpx.Client):
        """Verify different aspect ratios work."""
        for width, height in [(768, 512), (512, 768), (1024, 576)]:
            resp = client.post(
                "/generate",
                json={
                    "prompt": "A landscape",
                    "width": width,
                    "height": height,
                    "num_steps": 4,
                    "guidance": 1.0,
                    "seed": 1,
                },
            )
            assert resp.status_code == 200, f"Failed for {width}x{height}"
            data = resp.json()
            assert data["width"] == width
            assert data["height"] == height


class TestChallengingPoses:
    """Test challenging pose prompts that often cause body part issues."""

    @pytest.mark.parametrize("name,prompt", CHALLENGING_POSE_PROMPTS)
    def test_pose_generation_completes(self, client: httpx.Client, name: str, prompt: str):
        """Verify generation completes for challenging pose prompts.

        These prompts describe complex body positions that historically cause
        issues with limb generation (extra fingers, merged limbs, impossible joints).
        This test validates that generation completes without errors.
        Outputs are saved to output/test/ for manual review.
        """
        resp = client.post(
            "/generate",
            json={
                "prompt": prompt,
                "width": TEST_WIDTH,
                "height": TEST_HEIGHT,
                "num_steps": 50,
                "guidance": 4.0,
                "seed": 42,
            },
        )
        assert resp.status_code == 200, f"Generation failed for {name}: {resp.text}"
        data = resp.json()

        output_path = save_test_output(data["image_base64"], name.lower().replace(" ", "_"), 42)
        print(f"Saved: {output_path}")

        img_bytes = base64.b64decode(data["image_base64"])
        img = Image.open(BytesIO(img_bytes))
        assert img.size == (TEST_WIDTH, TEST_HEIGHT)
        assert not data.get("flagged", False), f"Output flagged for {name}"

    @pytest.mark.parametrize("name,prompt", CHALLENGING_POSE_PROMPTS)
    def test_pose_with_upsampling(self, client: httpx.Client, name: str, prompt: str):
        """Test challenging poses with prompt upsampling.

        Upsampling may help or hurt anatomical accuracy by adding more detail.
        Outputs saved with '_upsampled' suffix for comparison.
        """
        resp = client.post(
            "/generate",
            json={
                "prompt": prompt,
                "width": TEST_WIDTH,
                "height": TEST_HEIGHT,
                "num_steps": 50,
                "guidance": 4.0,
                "seed": 42,
                "upsample": True,
            },
        )
        assert resp.status_code == 200, f"Generation with upsample failed for {name}"
        data = resp.json()

        output_path = save_test_output(
            data["image_base64"], f"{name.lower().replace(' ', '_')}_upsampled", 42
        )
        print(f"Saved: {output_path}")
        print(f"Upsampled prompt: {data['prompt'][:200]}...")

        assert len(data["prompt"]) > len(prompt), "Upsampled prompt should be expanded"


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_prompt_accepted(self, client: httpx.Client):
        """Empty prompts are accepted (generate abstract output)."""
        resp = client.post(
            "/generate",
            json={
                "prompt": "",
                "width": 512,
                "height": 512,
                "num_steps": 4,
                "guidance": 1.0,
                "seed": 42,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "image_base64" in data

    def test_invalid_dimensions_rejected(self, client: httpx.Client):
        """Invalid dimensions should be rejected."""
        resp = client.post(
            "/generate",
            json={
                "prompt": "test",
                "width": 10,
                "height": 10,
                "num_steps": 4,
                "guidance": 1.0,
            },
        )
        assert resp.status_code == 422

    def test_deterministic_seed(self, client: httpx.Client):
        """Same seed should produce identical results."""
        payload = {
            "prompt": "A blue cube",
            "width": 256,
            "height": 256,
            "num_steps": 4,
            "guidance": 1.0,
            "seed": 12345,
        }
        resp1 = client.post("/generate", json=payload)
        resp2 = client.post("/generate", json=payload)

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        data1 = resp1.json()
        data2 = resp2.json()

        assert data1["image_base64"] == data2["image_base64"], "Same seed should produce identical output"
