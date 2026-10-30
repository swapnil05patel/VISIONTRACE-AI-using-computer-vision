import { Router, type IRouter } from "express";
import healthRouter from "./health";
import storageRouter from "./storage";
import visionTraceRouter from "./visiontrace";

const router: IRouter = Router();

router.use(healthRouter);
router.use(storageRouter);
router.use(visionTraceRouter);

export default router;
