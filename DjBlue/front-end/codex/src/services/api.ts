export type AuthResponse = {
  email: string;
  plan: string;
  token: string;
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const authService = {
  async login(email: string, password: string): Promise<AuthResponse> {
    await sleep(600);
    return {
      email,
      plan: "Crescendo",
      token: `mock-token-${Date.now()}`
    };
  },
  async signup(name: string, email: string, password: string): Promise<AuthResponse> {
    await sleep(800);
    return {
      email,
      plan: "Prelude",
      token: `mock-token-${Date.now()}`
    };
  }
};

export const subscriptionService = {
  async getStatus() {
    await sleep(400);
    return {
      plan: "Crescendo",
      renewalDate: "May 12, 2024",
      paymentMethod: "•• 2194 — BlueCard",
      seats: 3,
      addOns: ["Emotion analytics", "Creative collaborator"]
    };
  },
  async updatePlan(plan: string) {
    await sleep(500);
    return { plan, status: "updated" };
  }
};

export const downloadService = {
  async getBuilds() {
    await sleep(350);
    return [
      { platform: "macOS", version: "1.4.1", size: "196 MB", notes: "Optimized for Apple Silicon." },
      { platform: "Windows", version: "1.4.1", size: "204 MB", notes: "Improved audio routing." }
    ];
  }
};
